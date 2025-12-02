import { createBrowserRouter, RouterProvider, Outlet, isRouteErrorResponse, useRouteError } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { AuthProvider } from './contexts/AuthContext';
import RequireAuth from './components/RequireAuth';
import RequireGuest from './components/RequireGuest';

// Lazy load pages for code splitting
const Home = lazy(() => import('./pages/Home'));
const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./pages/Register'));
const ForgotPassword = lazy(() => import('./pages/ForgotPassword'));
const VerifyEmail = lazy(() => import('./pages/VerifyEmail'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const AuditLogs = lazy(() => import('./pages/AuditLogs'));
const TickManager = lazy(() => import('./pages/TickManager'));
const NotFound = lazy(() => import('./pages/NotFound'));

// Loading fallback
const Loading = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="text-lg text-gray-600">Loading...</div>
  </div>
);

// Error boundary for routes
const RouteError = () => {
  const error = useRouteError();
  let title = "Something went wrong";
  let details: string | undefined;

  if (isRouteErrorResponse(error)) {
    title = `${error.status} ${error.statusText || "Error"}`;
    try {
      details = typeof error.data === "string" ? error.data : JSON.stringify(error.data);
    } catch {}
  } else if (error instanceof Error) {
    title = error.name || title;
    details = error.message;
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6">
      <div className="max-w-md text-center">
        <h1 className="text-2xl font-bold text-red-600 mb-4">{title}</h1>
        {details && (
          <pre className="text-sm text-gray-600 bg-gray-100 p-4 rounded-lg overflow-auto max-h-64 text-left">
            {details}
          </pre>
        )}
        <p className="mt-4 text-gray-600">
          Try reloading the page or navigating back.
        </p>
      </div>
    </div>
  );
};

// Root layout component with auth provider
const RootLayout = () => {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-gray-50">
        <Suspense fallback={<Loading />}>
          <Outlet />
        </Suspense>
      </div>
    </AuthProvider>
  );
};

// Create router with auth guards
export const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    errorElement: <RouteError />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      // Guest-only routes (redirect to dashboard if authenticated)
      {
        element: <RequireGuest />,
        children: [
          {
            path: "sign-in",
            element: <Login />,
          },
          {
            path: "sign-up",
            element: <Register />,
          },
          {
            path: "forgot-password",
            element: <ForgotPassword />,
          },
        ],
      },
      // Public routes
      {
        path: "email/verify",
        element: <VerifyEmail />,
      },
      // Protected routes (require authentication)
      {
        element: <RequireAuth />,
        children: [
          {
            path: "dashboard",
            element: <Dashboard />,
          },
          {
            path: "audit-logs",
            element: <AuditLogs />,
          },
          {
            path: "tick-manager",
            element: <TickManager />,
          },
        ],
      },
      {
        path: "*",
        element: <NotFound />,
      },
    ],
  },
]);

// Router Provider Component
export function AppRouter() {
  return <RouterProvider router={router} />;
}
