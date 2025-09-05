import { supabase } from '../../lib/supabase';

export class Analytics {
  static async trackEvent(eventName: string, properties: Record<string, any> = {}) {
    try {
      const { data: user } = await supabase.auth.getUser();
      
      await supabase
        .from('analytics_events')
        .insert({
          user_id: user?.user?.id,
          event_name: eventName,
          properties,
          timestamp: new Date().toISOString(),
        });
    } catch (error) {
      console.error('Error tracking analytics event:', error);
    }
  }

  static async trackBrew(brewData: any) {
    return this.trackEvent('brew_logged', brewData);
  }

  static async trackCoffeeAdded(coffeeData: any) {
    return this.trackEvent('coffee_added', coffeeData);
  }
}

export default Analytics;