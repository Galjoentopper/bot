import React, { useState, useEffect } from 'react';
import { supabase, BrewLogWithCoffee } from '../lib/supabase';
import ExportShare from '../components/ExportShare';
import { formatDate } from '../utils/dateUtils';

const BrewLog: React.FC = () => {
  const [brews, setBrews] = useState<BrewLogWithCoffee[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadBrews();
  }, []);

  const loadBrews = async () => {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      const { data } = await supabase
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

      setBrews(data || []);
    } catch (error) {
      console.error('Error loading brews:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading brew log...</div>;
  }

  return (
    <div>
      <h1>Brew Log</h1>
      
      {brews.length === 0 ? (
        <p>No brews recorded yet. Start brewing!</p>
      ) : (
        <div className="brew-list">
          {brews.map((brew) => (
            <div key={brew.id} className="brew-item">
              <h3>{brew.coffee_bags?.coffee_templates?.name || 'Unknown Coffee'}</h3>
              <p><strong>Method:</strong> {brew.brewing_method}</p>
              <p><strong>Date:</strong> {formatDate(brew.brewed_at)}</p>
              <p><strong>Location:</strong> {brew.location}</p>
              {brew.rating && <p><strong>Rating:</strong> {brew.rating}/5</p>}
              {brew.notes && <p><strong>Notes:</strong> {brew.notes}</p>}
              
              <ExportShare 
                data={{
                  name: brew.coffee_bags?.coffee_templates?.name || 'Unknown',
                  origin: brew.coffee_bags?.coffee_templates?.origin || 'Unknown',
                  roast_level: brew.coffee_bags?.coffee_templates?.roast_level || 'Unknown',
                  ...brew
                }} 
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BrewLog;