import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { getUserAuditLogs, type AuditLog } from '../api/audit';

export default function AuditLogs() {
  const { user } = useAuth();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    if (!user) return;
    
    async function fetchAuditLogs() {
      if (!user) return;
      
      setLoading(true);
      setError(null);

      try {
        const data = await getUserAuditLogs(user.id, page, 20);
        setLogs(data.logs);
        setTotalPages(data.pages);
        setTotal(data.total);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch audit logs');
      } finally {
        setLoading(false);
      }
    }

    fetchAuditLogs();
  }, [user, page]);

  function formatDate(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleString();
  }

  function formatMetadata(metadata?: string): string {
    if (!metadata) return '-';
    try {
      const parsed = JSON.parse(metadata);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return metadata;
    }
  }

  if (!user) {
    return (
      <div className="p-6">
        <p className="text-gray-600">Please sign in to view audit logs</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Audit Logs</h1>
          <p className="text-gray-600">
            Activity history for {user.name}
            {total > 0 && ` (${total} total entries)`}
          </p>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center p-12">
            <div className="text-gray-600">Loading audit logs...</div>
          </div>
        ) : logs.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <p className="text-gray-600">No audit logs found</p>
          </div>
        ) : (
          <>
            <div className="bg-white rounded-lg shadow overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Timestamp
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Action
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Resource
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Details
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDate(log.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                          {log.action}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {log.resource_type && (
                          <div>
                            <div className="font-medium">{log.resource_type}</div>
                            {log.resource_id && (
                              <div className="text-gray-500 text-xs truncate max-w-xs">
                                {log.resource_id}
                              </div>
                            )}
                          </div>
                        )}
                        {!log.resource_type && <span className="text-gray-400">-</span>}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        <details className="cursor-pointer">
                          <summary className="text-blue-600 hover:text-blue-700">
                            View details
                          </summary>
                          <div className="mt-2 p-3 bg-gray-50 rounded text-xs">
                            <div className="grid grid-cols-2 gap-2">
                              {log.ip_address && (
                                <>
                                  <div className="font-medium">IP:</div>
                                  <div>{log.ip_address}</div>
                                </>
                              )}
                              {log.user_agent && (
                                <>
                                  <div className="font-medium">User Agent:</div>
                                  <div className="truncate">{log.user_agent}</div>
                                </>
                              )}
                              {log.metadata && (
                                <>
                                  <div className="font-medium col-span-2">Metadata:</div>
                                  <div className="col-span-2">
                                    <pre className="whitespace-pre-wrap">
                                      {formatMetadata(log.metadata)}
                                    </pre>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>
                        </details>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-6 flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  Page {page} of {totalPages}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
