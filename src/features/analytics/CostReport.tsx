import React, { useState, useEffect } from 'react';
import CostAnalysis from '../../components/CostAnalysis';
import { supabase } from '../../lib/supabase';

const CostReport: React.FC = () => {
  const [reportData, setReportData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCostData();
  }, []);

  const loadCostData = async () => {
    try {
      const { data } = await supabase
        .from('brew_logs')
        .select(`
          *,
          coffee_bags (
            *,
            coffee_templates (*)
          )
        `);

      setReportData(data || []);
    } catch (error) {
      console.error('Error loading cost data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading cost report...</div>;
  }

  return (
    <div>
      <h2>Cost Report</h2>
      <CostAnalysis data={reportData} />
    </div>
  );
};

export default CostReport;