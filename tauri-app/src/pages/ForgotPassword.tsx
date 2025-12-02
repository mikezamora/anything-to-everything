export default function ForgotPassword() {
  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8">
        <h1 className="text-2xl font-bold mb-4">Reset Password</h1>
        <p className="text-gray-600 mb-6">
          Enter your email to receive a password reset link.
        </p>
      </div>
    </div>
  );
}
