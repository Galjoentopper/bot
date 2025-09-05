import React from 'react';
import { ShareData, BrewLogWithCoffee } from '../types/database';

interface ExportShareProps {
  data?: ShareData | BrewLogWithCoffee;
}

const ExportShare: React.FC<ExportShareProps> = ({ data }) => {
  if (!data) {
    return <div>No data to share</div>;
  }

  // Handle both ShareData and BrewLogWithCoffee types
  let coffeeName = 'Unknown Coffee';
  
  if ('name' in data) {
    // ShareData type
    coffeeName = data.name;
  } else if ('coffee_bags' in data) {
    // BrewLogWithCoffee type
    coffeeName = data.coffee_bags?.coffee_templates?.name || 'Unknown Coffee';
  }

  return (
    <div className="export-share">
      <h3>Share Coffee Experience</h3>
      <p>Coffee: {coffeeName}</p>
      {/* Add more sharing functionality here */}
    </div>
  );
};

export default ExportShare;