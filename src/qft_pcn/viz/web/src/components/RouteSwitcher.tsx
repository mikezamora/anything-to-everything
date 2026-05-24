import { useVizStore } from '../store';

const ROUTES = [
  { key: 'viz',      label: 'Viz' },
  { key: 'dsl',      label: 'DSL' },
  { key: 'learn',    label: 'Learn' },
  { key: 'training', label: 'Training' },
] as const;

export function RouteSwitcher() {
  const route = useVizStore((s) => s.route);
  const setRoute = useVizStore((s) => s.setRoute);
  return (
    <div className="route-switcher">
      {ROUTES.map((r) => (
        <button
          key={r.key}
          type="button"
          className={r.key === route ? 'active' : ''}
          onClick={() => setRoute(r.key)}
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}
