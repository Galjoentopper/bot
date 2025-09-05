import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { formatDate } from '../utils/dateUtils';

interface CostReportData {
  date: string;
  time: string;
  coffee_name: string;
  brew_method: string;
  coffee_amount_g: number;
  water_amount_ml: number;
  grind_size: string;
  water_temp: number;
  brew_time_seconds: number;
  estimated_cost: number;
  rating: number;
  notes: string;
}

const CostReport: React.FC = () => {
  const [reportData, setReportData] = useState<CostReportData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    generateCostReport();
  }, []);

  const generateCostReport = async () => {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      const { data: logs } = await supabase
        .from('brew_logs')
        .select(`
          *,
          coffee_bags (
            *,
            coffee_templates (*)
          )
        `)
        .eq('user_id', user.user.id)
        .order('brewed_at', { ascending: false });

      if (!logs) return;

      const processedData: CostReportData[] = logs.map((log: any) => {
        // Calculate estimated cost
        let estimatedCost = log.estimated_cost;

        if (!estimatedCost && log.coffee_bags && log.coffee_amount) {
          const coffee = log.coffee_bags;
          if (coffee.price && coffee.weight) {
            let coffeeAmountInCostUnit = log.coffee_amount;
            if (coffee.unit_type === 'kg') coffeeAmountInCostUnit = log.coffee_amount / 1000;
            else if (coffee.unit_type === 'lb') coffeeAmountInCostUnit = log.coffee_amount / 453.592;
            else if (coffee.unit_type === 'oz') coffeeAmountInCostUnit = log.coffee_amount / 28.3495;

            // Calculate cost per gram/unit
            estimatedCost = (coffee.price / coffee.weight) * log.coffee_amount;
          }
        }

        return {
          date: new Date(log.brewed_at).toLocaleDateString(),
          time: new Date(log.brewed_at).toLocaleTimeString(),
          coffee_name: log.coffee_bags?.coffee_templates?.name || 'Unknown',
          brew_method: log.brewing_method || log.brew_method || 'Unknown',
          coffee_amount_g: log.coffee_amount,
          water_amount_ml: log.water_amount,
          grind_size: log.grind_size,
          water_temp: log.water_temp,
          brew_time_seconds: log.brew_time,
          estimated_cost: estimatedCost || 0,
          rating: log.rating,
          notes: log.notes
        };
      });

      setReportData(processedData);
    } catch (error) {
      console.error('Error generating cost report:', error);
    } finally {
      setLoading(false);
    }
  };

  const totalCost = reportData.reduce((sum, item) => sum + (item.estimated_cost || 0), 0);
  const averageCost = reportData.length > 0 ? totalCost / reportData.length : 0;

  if (loading) {
    return <div>Generating cost report...</div>;
  }

  return (
    <div>
      <h1>Cost Report</h1>
      
      <div className="cost-summary">
        <h2>Summary</h2>
        <p><strong>Total Brews:</strong> {reportData.length}</p>
        <p><strong>Total Cost:</strong> ${totalCost.toFixed(2)}</p>
        <p><strong>Average Cost per Brew:</strong> ${averageCost.toFixed(2)}</p>
      </div>

      <div className="cost-table">
        <h2>Detailed Report</h2>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Coffee</th>
              <th>Method</th>
              <th>Amount (g)</th>
              <th>Cost</th>
              <th>Rating</th>
            </tr>
          </thead>
          <tbody>
            {reportData.map((item, index) => (
              <tr key={index}>
                <td>{item.date}</td>
                <td>{item.coffee_name}</td>
                <td>{item.brew_method}</td>
                <td>{item.coffee_amount_g || 'N/A'}</td>
                <td>${(item.estimated_cost || 0).toFixed(2)}</td>
                <td>{item.rating || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CostReport;