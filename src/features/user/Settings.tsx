import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { ThemeToggle } from '../../components/ThemeToggle';
import { ThemeCustomizer } from '../../components/ThemeCustomizer';
import PWAInstall from '../../components/PWAInstall';
import { supabase } from '../../lib/supabase';

const Settings: React.FC = () => {
  const { theme } = useTheme();

  const handleExportData = async () => {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      // Export user data
      const { data: coffees } = await supabase
        .from('coffee_bags')
        .select('*')
        .eq('owner_id', user.user.id);

      const { data: brews } = await supabase
        .from('brew_logs')
        .select('*')
        .eq('user_id', user.user.id);

      const exportData = {
        coffees,
        brews,
        exported_at: new Date().toISOString(),
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json',
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `coffee-data-export-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting data:', error);
    }
  };

  return (
    <div className="settings">
      <h1>Settings</h1>

      <section>
        <h2>Theme</h2>
        <p>Current theme: {theme}</p>
        <ThemeToggle />
        <ThemeCustomizer />
      </section>

      <section>
        <h2>App Installation</h2>
        <PWAInstall />
      </section>

      <section>
        <h2>Data</h2>
        <button onClick={handleExportData}>
          Export My Data
        </button>
      </section>
    </div>
  );
};

export default Settings;