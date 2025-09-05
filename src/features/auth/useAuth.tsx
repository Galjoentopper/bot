import { useState, useEffect } from 'react';
import { supabase } from '../../lib/supabase';
// Mock User type since we don't have @supabase/supabase-js installed
interface User {
  id: string;
  email?: string;
  [key: string]: any;
}

interface AuthData {
  user: User | null;
  session?: any;
}
import { executeSupabaseQuery } from '../../utils/supabaseTimeout';

interface AuthState {
  user: User | null;
  loading: boolean;
}

export const useAuth = () => {
  const [state, setState] = useState<AuthState>({
    user: null,
    loading: true,
  });

  useEffect(() => {
    // Get initial user
    const getUser = async () => {
      try {
        const { data } = await executeSupabaseQuery(
          () => supabase.auth.getUser()
        );
        setState({ user: data.user, loading: false });
      } catch (error) {
        console.error('Error getting user:', error);
        setState({ user: null, loading: false });
      }
    };

    getUser();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        setState({ user: session?.user ?? null, loading: false });
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const signIn = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    return { data, error };
  };

  const signUp = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    });
    return { data, error };
  };

  const signOut = async () => {
    const { error } = await supabase.auth.signOut();
    return { error };
  };

  return {
    user: state.user,
    loading: state.loading,
    signIn,
    signUp,
    signOut,
  };
};