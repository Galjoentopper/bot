import React, { useState } from 'react';
import { supabase } from '../lib/supabase';
import { CreateCoffeeData } from '../types/database';

const AddCoffee: React.FC = () => {
  const [formData, setFormData] = useState<CreateCoffeeData>({
    name: '',
    origin: '',
    vendor: '',
    farm: '',
    roast_level: 'medium',
    processing_method: 'washed',
    price: 0,
    weight: 0,
    rating: 5,
    is_favorite: false,
  });
  const [photo, setPhoto] = useState<File | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      let photoUrl = null;

      // Upload photo if provided
      if (photo) {
        const fileExt = photo.name.split('.').pop();
        const fileName = `${Date.now()}.${fileExt}`;
        
        const { data: uploadData } = await supabase.storage
          .from('coffee-photos')
          .upload(fileName, photo);

        if (uploadData) {
          photoUrl = uploadData.path;
        }
      }

      // Create coffee template
      const templateData = {
        name: formData.name,
        origin: formData.origin,
        vendor: formData.vendor,
        farm: formData.farm,
        roast_level: formData.roast_level,
        processing_method: formData.processing_method,
        photo_url: photoUrl,
        created_by: user.user.id,
      };

      const { data: template } = await supabase
        .from('coffee_templates')
        .insert(templateData)
        .select()
        .single();

      if (!template) throw new Error('Failed to create template');

      let templateId = template.id;

      // Update template with photo URL if uploaded
      if (photoUrl && template) {
        await supabase
          .from('coffee_templates')
          .update({ photo_url: photoUrl })
          .eq('id', template.id);
      }

      // Create coffee bag
      const bagData = {
        template_id: templateId,
        owner_id: user.user.id,
        roast_date: new Date().toISOString(),
        purchase_date: new Date().toISOString(),
        price: formData.price || 0,
        weight: formData.weight || 0,
        rating: formData.rating || 5,
        is_favorite: formData.is_favorite || false,
      };

      await supabase
        .from('coffee_bags')
        .insert(bagData);

      alert('Coffee added successfully!');
      
      // Reset form
      setFormData({
        name: '',
        origin: '',
        vendor: '',
        farm: '',
        roast_level: 'medium',
        processing_method: 'washed',
        price: 0,
        weight: 0,
        rating: 5,
        is_favorite: false,
      });
      setPhoto(null);

    } catch (error) {
      console.error('Error adding coffee:', error);
      alert('Error adding coffee');
    }
  };

  return (
    <div>
      <h1>Add New Coffee</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Name:
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
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
              onChange={(e) => setFormData(prev => ({ ...prev, origin: e.target.value }))}
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
              onChange={(e) => setFormData(prev => ({ ...prev, vendor: e.target.value }))}
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
              onChange={(e) => setFormData(prev => ({ ...prev, farm: e.target.value }))}
              required
            />
          </label>
        </div>

        <div>
          <label>
            Photo:
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setPhoto(e.target.files?.[0] || null)}
            />
          </label>
        </div>

        <div>
          <label>
            Price:
            <input
              type="number"
              step="0.01"
              value={formData.price || 0}
              onChange={(e) => setFormData(prev => ({ ...prev, price: Number(e.target.value) }))}
            />
          </label>
        </div>

        <div>
          <label>
            Weight (g):
            <input
              type="number"
              value={formData.weight || 0}
              onChange={(e) => setFormData(prev => ({ ...prev, weight: Number(e.target.value) }))}
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
              value={formData.rating || 5}
              onChange={(e) => setFormData(prev => ({ ...prev, rating: Number(e.target.value) }))}
            />
          </label>
        </div>

        <div>
          <label>
            Favorite:
            <input
              type="checkbox"
              checked={formData.is_favorite || false}
              onChange={(e) => setFormData(prev => ({ ...prev, is_favorite: e.target.checked }))}
            />
          </label>
        </div>

        <button type="submit">Add Coffee</button>
      </form>
    </div>
  );
};

export default AddCoffee;