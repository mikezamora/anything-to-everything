import '@testing-library/jest-dom/vitest';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useVizStore } from '../../store';

vi.mock('@monaco-editor/react', () => ({
  default: ({ value, onChange }: any) => (
    <textarea aria-label="dsl-editor" value={value}
              onChange={(e) => onChange(e.target.value)} />
  ),
}));

import { DslEditor } from './DslEditor';

beforeEach(() => useVizStore.getState().resetAll());

describe('DslEditor', () => {
  it('renders the editor textarea bound to store.dslText', () => {
    useVizStore.getState().setDslText('{"hello":1}');
    render(<DslEditor />);
    const ta = screen.getByLabelText('dsl-editor') as HTMLTextAreaElement;
    expect(ta.value).toBe('{"hello":1}');
  });

  it('typing updates store.dslText', () => {
    render(<DslEditor />);
    const ta = screen.getByLabelText('dsl-editor') as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: '{"x":2}' } });
    expect(useVizStore.getState().dslText).toBe('{"x":2}');
  });

  it('shows validation OK when JSON parses', () => {
    useVizStore.getState().setDslText('{"fields":[]}');
    render(<DslEditor />);
    expect(screen.getByText(/JSON/i)).toBeInTheDocument();
  });
});
