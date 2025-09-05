import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { CoffeeBag } from '../types/database';

interface CoffeeDetailsProps {
  coffeeId: string;
}

const CoffeeDetails: React.FC<CoffeeDetailsProps> = ({ coffeeId }) => {
  const [bags, setBags] = useState<CoffeeBag[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCoffeeBags();
  }, [coffeeId]);

  const loadCoffeeBags = async () => {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      const { data } = await supabase
        .from('coffee_bags')
        .select(`
          *,
          coffee_templates (*)
        `)
        .eq('owner_id', user.user.id);

      setBags(data || []);
    } catch (error) {
      console.error('Error loading coffee bags:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading coffee details...</div>;
  }

  const bagIds = bags?.map(bag => bag.id) || [];

  return (
    <div>
      <h1>Coffee Collection</h1>
      <p>Total bags: {bagIds.length}</p>
      
      <div className="coffee-grid">
        {bags.map((bag) => (
          <div key={bag.id} className="coffee-card">
            <h3>{bag.coffee_templates?.name || 'Unknown Coffee'}</h3>
            <p><strong>Origin:</strong> {bag.coffee_templates?.origin}</p>
            <p><strong>Roast:</strong> {bag.coffee_templates?.roast_level}</p>
            {bag.price && <p><strong>Price:</strong> ${bag.price}</p>}
            {bag.weight && <p><strong>Weight:</strong> {bag.weight}g</p>}
            {bag.rating && <p><strong>Rating:</strong> {bag.rating}/5</p>}
            {bag.is_favorite && <span className="favorite">⭐ Favorite</span>}
          </div>
        ))}
      </div>
    </div>
  );
};

export default CoffeeDetails;