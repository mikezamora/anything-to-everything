import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LearnArticle } from './LearnArticle';

describe('LearnArticle', () => {
  it('renders the missing-article fallback when id is not registered', () => {
    render(<LearnArticle articleId="this-does-not-exist" />);
    expect(screen.getByText(/no article registered/i)).toBeInTheDocument();
  });
});
