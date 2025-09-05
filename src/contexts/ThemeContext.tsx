import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { supabase } from '../lib/supabase';
import { executeSupabaseQuery } from '../utils/supabaseTimeout';

export type Theme = 'light' | 'dark' | 'custom';

export interface CustomThemeColors {
  primary?: string;
  secondary?: string;
  accent?: string;
  background?: string;
  surface?: string;
  text?: string;
}

interface ThemeContextType {
  theme: Theme;
  customColors: CustomThemeColors;
  setTheme: (theme: Theme) => void;
  updateCustomColors: (colors: CustomThemeColors) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('light');
  const [customColors, setCustomColors] = useState<CustomThemeColors>({});

  useEffect(() => {
    loadUserTheme();
  }, []);

  const loadUserTheme = async () => {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      const result = await executeSupabaseQuery(
        () => supabase
          .from('user_preferences')
          .select('theme, custom_theme_colors')
          .eq('user_id', user.user!.id)
          .single()
      );

      if (result.data) {
        setTheme(result.data.theme || 'light');
        setCustomColors(result.data.custom_theme_colors || {});
      }
    } catch (error) {
      console.error('Error loading user theme:', error);
    }
  };

  const updateTheme = async (newTheme: Theme) => {
    setTheme(newTheme);
    
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      await executeSupabaseQuery(
        () => supabase
          .from('user_preferences')
          .upsert({
            user_id: user.user!.id,
            theme: newTheme,
            updated_at: new Date().toISOString(),
          })
      );
    } catch (error) {
      console.error('Error saving theme:', error);
    }
  };

  const updateCustomColors = async (colors: CustomThemeColors) => {
    setCustomColors(colors);
    
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      await executeSupabaseQuery(
        () => supabase
          .from('user_preferences')
          .upsert({
            user_id: user.user!.id,
            custom_theme_colors: colors,
            updated_at: new Date().toISOString(),
          })
      );
    } catch (error) {
      console.error('Error saving custom colors:', error);
    }
  };

  const saveThemeAndColors = async (newTheme: Theme, colors: CustomThemeColors) => {
    setTheme(newTheme);
    setCustomColors(colors);
    
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      await executeSupabaseQuery(
        () => supabase
          .from('user_preferences')
          .upsert({
            user_id: user.user!.id,
            theme: newTheme,
            custom_theme_colors: colors,
            updated_at: new Date().toISOString(),
          })
      );
    } catch (error) {
      console.error('Error saving theme and colors:', error);
    }
  };

  const value: ThemeContextType = {
    theme,
    customColors,
    setTheme: updateTheme,
    updateCustomColors,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};