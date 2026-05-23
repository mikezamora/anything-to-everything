/** A horizontal strip of toggle buttons rendered above a panel's main viz. */

export interface ToolbarItem {
  key: string;
  label: string;
  active: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

export function PanelToolbar({ items }: { items: ToolbarItem[] }) {
  return (
    <div className="panel-toolbar">
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          className={`panel-toolbar-btn${it.active ? ' active' : ''}`}
          onClick={it.onToggle}
          disabled={it.disabled}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}
