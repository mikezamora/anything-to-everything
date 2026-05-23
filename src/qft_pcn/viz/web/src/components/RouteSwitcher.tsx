import { useVizStore } from '../store';

export function RouteSwitcher() {
  const route = useVizStore((s) => s.route);
  const setRoute = useVizStore((s) => s.setRoute);
  return (
    <div className="route-switcher">
      {(['viz', 'dsl'] as const).map((r) => (
        <button
          key={r}
          type="button"
          className={r === route ? 'active' : ''}
          onClick={() => setRoute(r)}
        >
          {r === 'viz' ? 'Viz' : 'DSL'}
        </button>
      ))}
    </div>
  );
}
