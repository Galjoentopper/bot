import React from 'react';

// Coffee icon component as a fallback
export const Coffee: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M8 13v-1a4 4 0 014-4 4 4 0 014 4v1M8 13h8M8 13l-1 7h10l-1-7M3 9h2l1-7h12l1 7h2"
    />
  </svg>
);

export const getCoffeeIcon = (iconName?: string) => {
  // Return default coffee icon for any request
  return Coffee;
};