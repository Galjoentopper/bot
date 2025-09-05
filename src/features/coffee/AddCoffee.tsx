import React, { useState } from 'react';
import { useCoffee } from '../../hooks/useCoffee';
import { CreateCoffeeData } from '../../types/database';

const AddCoffee: React.FC = () => {
  const { addCoffee } = useCoffee();
  const [formData, setFormData] = useState<CreateCoffeeData>({
    name: '',
    origin: '',
    vendor: '',
    farm: '',
    roast_level: 'medium',
    processing_method: 'washed',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await addCoffee(formData);
      // Reset form or navigate away
      setFormData({
        name: '',
        origin: '',
        vendor: '',
        farm: '',
        roast_level: 'medium',
        processing_method: 'washed',
      });
    } catch (error) {
      console.error('Error adding coffee:', error);
    }
  };

  const handleChange = (field: keyof CreateCoffeeData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Add New Coffee</h2>
      
      <div>
        <label>
          Name:
          <input
            type="text"
            value={formData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            required
          />
        </label>
      </div>

      <div>
        <label>
          Origin:
          <input
            type="text"
            value={formData.origin}
            onChange={(e) => handleChange('origin', e.target.value)}
            required
          />
        </label>
      </div>

      <div>
        <label>
          Vendor:
          <input
            type="text"
            value={formData.vendor}
            onChange={(e) => handleChange('vendor', e.target.value)}
            required
          />
        </label>
      </div>

      <div>
        <label>
          Farm:
          <input
            type="text"
            value={formData.farm}
            onChange={(e) => handleChange('farm', e.target.value)}
            required
          />
        </label>
      </div>

      <div>
        <label>
          Roast Level:
          <select
            value={formData.roast_level}
            onChange={(e) => handleChange('roast_level', e.target.value as any)}
          >
            <option value="light">Light</option>
            <option value="medium-light">Medium Light</option>
            <option value="medium">Medium</option>
            <option value="medium-dark">Medium Dark</option>
            <option value="dark">Dark</option>
          </select>
        </label>
      </div>

      <div>
        <label>
          Processing Method:
          <select
            value={formData.processing_method}
            onChange={(e) => handleChange('processing_method', e.target.value as any)}
          >
            <option value="natural">Natural</option>
            <option value="washed">Washed</option>
            <option value="honey">Honey</option>
            <option value="semi-washed">Semi-washed</option>
            <option value="wet-hulled">Wet-hulled</option>
            <option value="anaerobic">Anaerobic</option>
            <option value="carbonic-maceration">Carbonic Maceration</option>
          </select>
        </label>
      </div>

      <button type="submit">Add Coffee</button>
    </form>
  );
};

export default AddCoffee;