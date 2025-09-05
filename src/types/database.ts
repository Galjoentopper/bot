export interface Coffee {
  id: string;
  name: string;
  origin: string;
  vendor: string;
  farm: string;
  roast_level: 'light' | 'medium-light' | 'medium' | 'medium-dark' | 'dark';
  processing_method: 'natural' | 'washed' | 'honey' | 'semi-washed' | 'wet-hulled' | 'anaerobic' | 'carbonic-maceration';
  created_at: string;
  updated_at: string;
  created_by: string;
  coffee_templates?: CoffeeTemplate;
  price?: number;
  weight?: number;
  rating?: number;
  is_favorite?: boolean;
}

export interface CoffeeTemplate {
  id: string;
  name: string;
  origin: string;
  vendor: string;
  farm: string;
  roast_level: 'light' | 'medium-light' | 'medium' | 'medium-dark' | 'dark';
  processing_method: 'natural' | 'washed' | 'honey' | 'semi-washed' | 'wet-hulled' | 'anaerobic' | 'carbonic-maceration';
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface CoffeeBag {
  id: string;
  template_id: string;
  owner_id: string;
  price?: number | null;
  weight?: number | null;
  unit_type?: string | null;
  roast_date?: string | null;
  purchase_date?: string | null;
  rating?: number | null;
  is_favorite: boolean;
  status?: string | null;
  created_at: string;
  updated_at: string;
  coffee_templates?: CoffeeTemplate;
}

export interface BrewLog {
  id: string;
  user_id: string;
  coffee_bag_id: string;
  brewing_method: string;
  location: string;
  grind_setting?: number | null;
  water_temp?: number | null;
  brew_time?: number | null;
  rating?: number | null;
  notes?: string | null;
  brewed_at: string;
  created_at: string;
  updated_at?: string;
  coffee_amount?: number;
  water_amount?: number;
  grind_size?: string;
  estimated_cost?: number;
}

export interface BrewLogWithCoffee extends BrewLog {
  coffee_bags: CoffeeBag;
}

export interface ShareData {
  name: string;
  origin: string;
  roast_level: string;
  [key: string]: any;
}

export interface CreateCoffeeData {
  name: string;
  origin: string;
  vendor: string;
  farm: string;
  roast_level: 'light' | 'medium-light' | 'medium' | 'medium-dark' | 'dark';
  processing_method: 'natural' | 'washed' | 'honey' | 'semi-washed' | 'wet-hulled' | 'anaerobic' | 'carbonic-maceration';
  price?: number;
  weight?: number;
  rating?: number;
  is_favorite?: boolean;
}

export interface UseCoffeeReturn {
  addCoffee: (data: CreateCoffeeData) => Promise<void>;
  updateCoffee: (id: string, data: Partial<CreateCoffeeData>) => Promise<void>;
  deleteCoffee: (id: string) => Promise<void>;
  toggleFavorite: (id: string) => Promise<void>;
  // Add other methods as needed
}