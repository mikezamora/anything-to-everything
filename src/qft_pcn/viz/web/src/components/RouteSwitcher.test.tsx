import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RouteSwitcher } from './RouteSwitcher';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('RouteSwitcher', () => {
  it('renders all four route buttons', () => {
    render(<RouteSwitcher />);
    expect(screen.getByRole('button', { name: 'Viz' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'DSL' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Learn' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Training' })).toBeInTheDocument();
  });
  it('clicking DSL sets route in store', () => {
    render(<RouteSwitcher />);
    fireEvent.click(screen.getByRole('button', { name: 'DSL' }));
    expect(useVizStore.getState().route).toBe('dsl');
  });
  it('clicking Learn sets route to learn', () => {
    render(<RouteSwitcher />);
    fireEvent.click(screen.getByRole('button', { name: 'Learn' }));
    expect(useVizStore.getState().route).toBe('learn');
  });
});
