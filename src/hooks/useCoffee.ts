import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { CreateCoffeeData, UseCoffeeReturn, CoffeeBag, CoffeeTemplate } from '../types/database';
import { executeSupabaseQuery } from '../utils/supabaseTimeout';

export const useCoffee = (): UseCoffeeReturn => {
  const [loading, setLoading] = useState(false);

  const addCoffee = async (data: CreateCoffeeData): Promise<void> => {
    setLoading(true);
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) throw new Error('User not authenticated');

      // First, create or find the template
      const templateData = {
        name: data.name,
        origin: data.origin,
        vendor: data.vendor,
        farm: data.farm,
        roast_level: data.roast_level,
        processing_method: data.processing_method,
        created_by: user.user.id,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      const templateResult = await executeSupabaseQuery(
        () => supabase
          .from('coffee_templates')
          .insert(templateData)
          .select()
          .single()
      );

      if (templateResult.error) throw templateResult.error;
      const template = templateResult.data;

      // Create the coffee bag
      const bagData = {
        user_id: user.user.id,
        template_id: template.id,
        owner_id: user.user.id,
        price: data.price || null,
        weight: data.weight || null,
        unit_type: 'g',
        roast_date: new Date().toISOString(),
        purchase_date: new Date().toISOString(),
        rating: data.rating || null,
        is_favorite: data.is_favorite || false,
        status: 'active',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      await executeSupabaseQuery(
        () => supabase
          .from('coffee_bags')
          .insert(bagData)
      );

    } catch (error) {
      console.error('Error adding coffee:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const updateCoffee = async (id: string, data: Partial<CreateCoffeeData>): Promise<void> => {
    setLoading(true);
    try {
      await executeSupabaseQuery(
        () => supabase
          .from('coffee_bags')
          .update({
            ...data,
            updated_at: new Date().toISOString(),
          })
          .eq('id', id)
      );
    } catch (error) {
      console.error('Error updating coffee:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const deleteCoffee = async (id: string): Promise<void> => {
    setLoading(true);
    try {
      await executeSupabaseQuery(
        () => supabase
          .from('coffee_bags')
          .delete()
          .eq('id', id)
      );
    } catch (error) {
      console.error('Error deleting coffee:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const toggleFavorite = async (id: string): Promise<void> => {
    setLoading(true);
    try {
      // First get current coffee bag to toggle favorite
      const { data: currentBag } = await executeSupabaseQuery(
        () => supabase
          .from('coffee_bags')
          .select('is_favorite')
          .eq('id', id)
          .single()
      );

      if (!currentBag) throw new Error('Coffee bag not found');

      await executeSupabaseQuery(
        () => supabase
          .from('coffee_bags')
          .update({
            is_favorite: !currentBag.is_favorite,
            updated_at: new Date().toISOString(),
          })
          .eq('id', id)
      );
    } catch (error) {
      console.error('Error toggling favorite:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return {
    addCoffee,
    updateCoffee,
    deleteCoffee,
    toggleFavorite,
  };
};