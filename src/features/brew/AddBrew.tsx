import React, { useState } from 'react';
import { useBrew } from '../../hooks/useBrew';

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
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      await addBrew({
        user_id: '', // Will be set in the hook
        ...formData
      });
      
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
      });

      alert('Brew logged successfully!');
    } catch (error) {
      console.error('Error adding brew:', error);
      alert('Error adding brew');
    }
  };

  return (
    <div>
      <h2>Add New Brew</h2>
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