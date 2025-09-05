import React from 'react';
import { useTheme, CustomThemeColors } from '../contexts/ThemeContext';

export const ThemeCustomizer: React.FC = () => {
  const { customColors, updateCustomColors } = useTheme();

  const handleColorChange = (key: keyof CustomThemeColors, value: string) => {
    updateCustomColors({
      ...customColors,
      [key]: value,
    });
  };

  return (
    <div>
      <h3>Customize Theme</h3>
      <div>
        <label>
          Primary Color:
          <input
            type="color"
            value={customColors.primary || '#000000'}
            onChange={(e) => handleColorChange('primary', e.target.value)}
          />
        </label>
      </div>
      <div>
        <label>
          Background Color:
          <input
            type="color"
            value={customColors.background || '#ffffff'}
            onChange={(e) => handleColorChange('background', e.target.value)}
          />
        </label>
      </div>
      {/* Add more color customizations as needed */}
    </div>
  );
};