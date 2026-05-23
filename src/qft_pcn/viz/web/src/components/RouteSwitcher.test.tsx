import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RouteSwitcher } from './RouteSwitcher';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('RouteSwitcher', () => {
  it('renders two route buttons', () => {
    render(<RouteSwitcher />);
    expect(screen.getByRole('button', { name: 'Viz' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'DSL' })).toBeInTheDocument();
  });
  it('clicking DSL sets route in store', () => {
    render(<RouteSwitcher />);
    fireEvent.click(screen.getByRole('button', { name: 'DSL' }));
    expect(useVizStore.getState().route).toBe('dsl');
  });
});
