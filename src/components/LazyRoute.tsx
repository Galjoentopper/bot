import React, { Suspense, lazy, ComponentType } from 'react';
import ErrorBoundary from './ErrorBoundary';

interface ErrorFallbackProps {
  error: Error;
  retry: () => void;
}

const CustomErrorFallback: React.FC<ErrorFallbackProps> = ({ error, retry }) => (
  <div>
    <h2>Failed to load component</h2>
    <p>{error.message}</p>
    <button onClick={retry}>Retry</button>
  </div>
);

interface LazyRouteProps {
  component: () => Promise<{ default: ComponentType<any> }>;
  props?: any;
  loading?: React.ComponentType;
}

function LazyRoute({ 
  component, 
  props, 
  loading: LoadingComponent 
}: LazyRouteProps) {
  const LazyComponent = lazy(component);

  const retry = () => {
    window.location.reload();
  };

  const LoadingFallback = LoadingComponent || (() => <div>Loading...</div>);

  return (
    <ErrorBoundary fallback={<CustomErrorFallback error={new Error('Component failed to load')} retry={retry} />}>
      <Suspense fallback={<LoadingFallback />}>
        <LazyComponent {...(props || {})} />
      </Suspense>
    </ErrorBoundary>
  );
}

export default LazyRoute;