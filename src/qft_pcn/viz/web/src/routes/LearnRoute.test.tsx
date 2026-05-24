import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { LearnRoute } from './LearnRoute';

describe('LearnRoute', () => {
  it('mounts without crashing even when the articles registry is empty', () => {
    const { container } = render(<LearnRoute />);
    expect(container.querySelector('.learn-route')).toBeInTheDocument();
  });
});
