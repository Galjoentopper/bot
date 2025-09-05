import { useState } from 'react';
import { useAuth } from '../auth/useAuth';
import { supabase } from '../../lib/supabase';
import { executeSupabaseQuery } from '../../utils/supabaseTimeout';

interface UserProfile {
  id: string;
  email?: string;
  full_name?: string;
  preferences?: any;
}

export const useUserProfile = () => {
  const { user } = useAuth();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(false);

  const loadProfile = async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const result = await executeSupabaseQuery(
        () => supabase
          .from('user_profiles')
          .select('*')
          .eq('id', user.id)
          .single()
      );

      if (result.data) {
        setProfile(result.data);
      }
    } catch (error) {
      console.error('Error loading user profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateProfile = async (updates: Partial<UserProfile>) => {
    if (!user) return;

    setLoading(true);
    try {
      await executeSupabaseQuery(
        () => supabase
          .from('user_profiles')
          .upsert({
            id: user.id,
            ...updates,
            updated_at: new Date().toISOString(),
          })
      );

      setProfile(prev => prev ? { ...prev, ...updates } : null);
    } catch (error) {
      console.error('Error updating profile:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return {
    profile,
    loading,
    loadProfile,
    updateProfile,
  };
};