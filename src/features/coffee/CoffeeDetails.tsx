import React, { useState, useEffect } from 'react';
import { Coffee } from '../../components/CoffeeIcons';
import { Coffee as CoffeeType, BrewLog } from '../../types/database';
import { formatDate } from '../../utils/dateUtils';
import { supabase } from '../../lib/supabase';

interface CoffeeDetailsProps {
  coffeeId: string;
}

const CoffeeDetails: React.FC<CoffeeDetailsProps> = ({ coffeeId }) => {
  const [coffee, setCoffee] = useState<CoffeeType | null>(null);
  const [brews, setBrews] = useState<BrewLog[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCoffeeDetails();
  }, [coffeeId]);

  const loadCoffeeDetails = async () => {
    try {
      // Load coffee details
      const { data: coffeeData } = await supabase
        .from('coffee_bags')
        .select(`
          *,
          coffee_templates (*)
        `)
        .eq('id', coffeeId)
        .single();

      setCoffee(coffeeData);

      // Load associated brews
      const { data: brewData } = await supabase
        .from('brew_logs')
        .select('*')
        .eq('coffee_bag_id', coffeeId)
        .order('brewed_at', { ascending: false });

      setBrews(brewData || []);
    } catch (error) {
      console.error('Error loading coffee details:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <Coffee className="h-6 w-6 animate-spin text-amber-600" />
        <p>Loading coffee details...</p>
      </div>
    );
  }

  if (!coffee) {
    return <div>Coffee not found</div>;
  }

  return (
    <div className="coffee-details">
      <div className="coffee-header">
        <Coffee className="h-5 w-5 text-gray-500" />
        <h1>{coffee.coffee_templates?.name || 'Unknown Coffee'}</h1>
      </div>

      <div className="coffee-info">
        <p><strong>Origin:</strong> {coffee.coffee_templates?.origin}</p>
        <p><strong>Vendor:</strong> {coffee.coffee_templates?.vendor}</p>
        <p><strong>Farm:</strong> {coffee.coffee_templates?.farm}</p>
        <p><strong>Roast Level:</strong> {coffee.coffee_templates?.roast_level}</p>
        <p><strong>Processing:</strong> {coffee.coffee_templates?.processing_method}</p>
        {coffee.price && <p><strong>Price:</strong> ${coffee.price}</p>}
        {coffee.weight && <p><strong>Weight:</strong> {coffee.weight}g</p>}
        {coffee.rating && <p><strong>Rating:</strong> {coffee.rating}/5</p>}
      </div>

      <div className="brew-history">
        <h3>
          <Coffee className="h-5 w-5 text-gray-500" />
          Brew History
        </h3>

        {brews.length === 0 ? (
          <div className="no-brews">
            <Coffee className="h-8 w-8 text-gray-400 mx-auto mb-2" />
            <p>No brews recorded yet</p>
          </div>
        ) : (
          <div className="brew-list">
            {brews.map((brew) => (
              <div key={brew.id} className="brew-item">
                <div className="brew-method">
                  {brew.brewing_method}
                </div>
                <div className="brew-date">
                  {formatDate(brew.brewed_at)}
                </div>
                {brew.rating && (
                  <div className="brew-rating">
                    Rating: {brew.rating}/5
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="coffee-metadata">
        <p><strong>Added:</strong> {formatDate(coffee.created_at)}</p>
        {coffee.updated_at && coffee.updated_at !== coffee.created_at && (
          <p>
            <strong>Last Updated:</strong>{' '}
            {formatDate(coffee.updated_at)}
          </p>
        )}
      </div>
    </div>
  );
};

export default CoffeeDetails;