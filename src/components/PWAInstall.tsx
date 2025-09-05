import React from 'react';

const PWAInstall: React.FC = () => {
  const handleInstall = () => {
    // PWA installation logic would go here
    console.log('PWA install triggered');
  };

  return (
    <div>
      <h3>Install App</h3>
      <button onClick={handleInstall}>
        Install as App
      </button>
      <p>Install this app on your device for a better experience!</p>
    </div>
  );
};

export default PWAInstall;