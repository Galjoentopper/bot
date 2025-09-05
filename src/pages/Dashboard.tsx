import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

interface DashboardStats {
  totalCoffees: number;
  totalBrews: number;
  favoriteCount: number;
  averageRating: number;
}

const withTimeoutAndRetry = async <T,>(
  operation: () => Promise<T>,
  timeoutMs: number = 10000,
  retries: number = 3
): Promise<T> => {
  for (let i = 0; i < retries; i++) {
    try {
      return await Promise.race([
        operation(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Operation timed out')), timeoutMs)
        ),
      ]);
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
  throw new Error('Max retries exceeded');
};

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats>({
    totalCoffees: 0,
    totalBrews: 0,
    favoriteCount: 0,
    averageRating: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardStats();
  }, []);

  const loadDashboardStats = async () => {
    try {
      const { data: user } = await supabase.auth.getUser();
      if (!user?.user) return;

      // Get coffee count
      const { count: totalCoffees } = await withTimeoutAndRetry(() =>
        supabase
          .from('coffee_bags')
          .select('*', { count: 'exact', head: true })
          .eq('owner_id', user.user!.id)
      );

      // Get brew count
      const { count: totalBrews } = await withTimeoutAndRetry(() =>
        supabase
          .from('brew_logs')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', user.user!.id)
      );

      // Get favorites count
      const { count: favoriteCount } = await withTimeoutAndRetry(() =>
        supabase
          .from('coffee_bags')
          .select('*', { count: 'exact', head: true })
          .eq('owner_id', user.user!.id)
          .eq('is_favorite', true)
      );

      // Get average rating
      const { data: ratings } = await withTimeoutAndRetry(() =>
        supabase
          .from('coffee_bags')
          .select('rating')
          .eq('owner_id', user.user!.id)
          .not('rating', 'is', null)
      );

      const averageRating = ratings && ratings.length > 0
        ? ratings.reduce((sum: number, item: any) => sum + (item.rating || 0), 0) / ratings.length
        : 0;

      setStats({
        totalCoffees: totalCoffees || 0,
        totalBrews: totalBrews || 0,
        favoriteCount: favoriteCount || 0,
        averageRating: Math.round(averageRating * 10) / 10,
      });
    } catch (error) {
      console.error('Error loading dashboard stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading dashboard...</div>;
  }

  return (
    <div className="dashboard">
      <h1>Coffee Dashboard</h1>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Total Coffees</h3>
          <p className="stat-number">{stats.totalCoffees}</p>
        </div>
        
        <div className="stat-card">
          <h3>Total Brews</h3>
          <p className="stat-number">{stats.totalBrews}</p>
        </div>
        
        <div className="stat-card">
          <h3>Favorites</h3>
          <p className="stat-number">{stats.favoriteCount}</p>
        </div>
        
        <div className="stat-card">
          <h3>Average Rating</h3>
          <p className="stat-number">{stats.averageRating}/5</p>
        </div>
      </div>

      <div className="quick-actions">
        <h2>Quick Actions</h2>
        <button>Add New Coffee</button>
        <button>Log New Brew</button>
        <button>View Reports</button>
      </div>
    </div>
  );
};

export default Dashboard;