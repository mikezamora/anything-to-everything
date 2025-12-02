export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-4xl font-bold mb-4">Welcome to Anything-to-Everything</h1>
      <p className="text-lg text-gray-600 mb-8 text-center max-w-2xl">
        A powerful Tauri application with WASM plugin system for extensible functionality.
      </p>
      <div className="flex gap-4">
        <a 
          href="/sign-in" 
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          Sign In
        </a>
        <a 
          href="/sign-up" 
          className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition"
        >
          Sign Up
        </a>
      </div>
    </div>
  );
}
