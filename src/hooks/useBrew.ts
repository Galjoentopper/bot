import { useState } from 'react';
import { supabase } from '../lib/supabase';
import { BrewLog } from '../types/database';
import { executeSupabaseQuery } from '../utils/supabaseTimeout';

interface BrewData {
  id?: string;
  user_id: string;
  coffee_bag_id: string;
  brewing_method: string;
  location: string;
  grind_setting?: number;
  water_temp?: number;
  brew_time?: number;
  rating?: number;
  notes?: string;
  brewed_at?: string;
  created_at?: string;
}

export const useBrew = () => {
  const [loading, setLoading] = useState(false);

  const addBrew = async (brewData: BrewData): Promise<void> => {
    setLoading(true);
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) throw new Error('User not authenticated');

      const brewRecord = {
        user_id: user.user.id,
        coffee_bag_id: brewData.coffee_bag_id,
        brewing_method: brewData.brewing_method,
        location: brewData.location,
        grind_setting: brewData.grind_setting || null,
        water_temp: brewData.water_temp || null,
        brew_time: brewData.brew_time || null,
        rating: brewData.rating || null,
        notes: brewData.notes || null,
        brewed_at: brewData.brewed_at || new Date().toISOString(),
        created_at: new Date().toISOString(),
      };

      await executeSupabaseQuery(
        () => supabase
          .from('brew_logs')
          .insert(brewRecord)
      );
    } catch (error) {
      console.error('Error adding brew:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const updateBrew = async (id: string, updates: Partial<BrewData>): Promise<void> => {
    setLoading(true);
    try {
      await executeSupabaseQuery(
        () => supabase
          .from('brew_logs')
          .update({
            ...updates,
            updated_at: new Date().toISOString()
          })
          .eq('id', id)
      );
    } catch (error) {
      console.error('Error updating brew:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const deleteBrew = async (id: string): Promise<void> => {
    setLoading(true);
    try {
      await executeSupabaseQuery(
        () => supabase
          .from('brew_logs')
          .delete()
          .eq('id', id)
      );
    } catch (error) {
      console.error('Error deleting brew:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return {
    addBrew,
    updateBrew,
    deleteBrew,
    loading,
  };
};