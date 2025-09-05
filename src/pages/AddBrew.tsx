import React, { useState } from 'react';
import { useBrew } from '../hooks/useBrew';
import { supabase } from '../lib/supabase';

const AddBrew: React.FC = () => {
  const { addBrew } = useBrew();
  const [formData, setFormData] = useState({
    coffee_bag_id: '',
    brewing_method: '',
    location: '',
    grind_setting: 0,
    water_temp: 0,
    brew_time: 0,
    rating: 5,
    notes: '',
    bag_rating: 5,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      const brewData = {
        user_id: user.user.id,
        coffee_bag_id: formData.coffee_bag_id,
        brewing_method: formData.brewing_method,
        location: formData.location,
        grind_setting: formData.grind_setting,
        water_temp: formData.water_temp,
        brew_time: formData.brew_time,
        rating: formData.rating,
        notes: formData.notes,
      };

      // Insert brew log
      await supabase
        .from('brew_logs')
        .insert([brewData]);

      // Update coffee bag rating
      if (formData.bag_rating !== 5) {
        await supabase
          .from('coffee_bags')
          .update({ rating: formData.bag_rating })
          .eq('id', formData.coffee_bag_id);
      }

      // Reset form
      setFormData({
        coffee_bag_id: '',
        brewing_method: '',
        location: '',
        grind_setting: 0,
        water_temp: 0,
        brew_time: 0,
        rating: 5,
        notes: '',
        bag_rating: 5,
      });

      alert('Brew logged successfully!');
    } catch (error) {
      console.error('Error adding brew:', error);
      alert('Error adding brew');
    }
  };

  return (
    <div>
      <h1>Add New Brew</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Coffee Bag ID:
            <input
              type="text"
              value={formData.coffee_bag_id}
              onChange={(e) => setFormData(prev => ({ ...prev, coffee_bag_id: e.target.value }))}
              required
            />
          </label>
        </div>

        <div>
          <label>
            Brewing Method:
            <input
              type="text"
              value={formData.brewing_method}
              onChange={(e) => setFormData(prev => ({ ...prev, brewing_method: e.target.value }))}
              required
            />
          </label>
        </div>

        <div>
          <label>
            Location:
            <input
              type="text"
              value={formData.location}
              onChange={(e) => setFormData(prev => ({ ...prev, location: e.target.value }))}
              required
            />
          </label>
        </div>

        <div>
          <label>
            Grind Setting:
            <input
              type="number"
              value={formData.grind_setting}
              onChange={(e) => setFormData(prev => ({ ...prev, grind_setting: Number(e.target.value) }))}
            />
          </label>
        </div>

        <div>
          <label>
            Water Temperature (°C):
            <input
              type="number"
              value={formData.water_temp}
              onChange={(e) => setFormData(prev => ({ ...prev, water_temp: Number(e.target.value) }))}
            />
          </label>
        </div>

        <div>
          <label>
            Brew Time (seconds):
            <input
              type="number"
              value={formData.brew_time}
              onChange={(e) => setFormData(prev => ({ ...prev, brew_time: Number(e.target.value) }))}
            />
          </label>
        </div>

        <div>
          <label>
            Rating (1-5):
            <input
              type="number"
              min="1"
              max="5"
              value={formData.rating}
              onChange={(e) => setFormData(prev => ({ ...prev, rating: Number(e.target.value) }))}
            />
          </label>
        </div>

        <div>
          <label>
            Notes:
            <textarea
              value={formData.notes}
              onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
            />
          </label>
        </div>

        <button type="submit">Log Brew</button>
      </form>
    </div>
  );
};

export default AddBrew;