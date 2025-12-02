import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  async function handleSignOut() {
    try {
      await signOut();
      navigate('/', { replace: true });
    } catch (error) {
      console.error('Sign out failed:', error);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
            {user && (
              <p className="text-gray-600">
                Welcome back, {user.name}!
              </p>
            )}
          </div>
          <button
            onClick={handleSignOut}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
          >
            Sign Out
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">User Information</h2>
            {user && (
              <dl className="space-y-2">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Username</dt>
                  <dd className="text-base text-gray-900">{user.name}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Email</dt>
                  <dd className="text-base text-gray-900">{user.email}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Email Verified</dt>
                  <dd className="text-base text-gray-900">
                    {user.emailVerified ? (
                      <span className="text-green-600">✓ Verified</span>
                    ) : (
                      <span className="text-orange-600">⚠ Not verified</span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Member Since</dt>
                  <dd className="text-base text-gray-900">
                    {new Date(user.createdAt).toLocaleDateString()}
                  </dd>
                </div>
              </dl>
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Quick Links</h2>
            <div className="space-y-3">
              <Link
                to="/audit-logs"
                className="block p-3 rounded-lg border border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-colors"
              >
                <div className="font-medium text-gray-900">Audit Logs</div>
                <div className="text-sm text-gray-600">View your activity history and audit logs</div>
              </Link>
              <Link
                to="/tick-manager"
                className="block p-3 rounded-lg border border-gray-200 hover:border-green-500 hover:bg-green-50 transition-colors"
              >
                <div className="font-medium text-gray-900">Tick Manager</div>
                <div className="text-sm text-gray-600">Monitor and control the game tick system</div>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
