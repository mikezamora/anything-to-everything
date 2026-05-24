import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LearnArticle } from './LearnArticle';
import { ARTICLES } from './articles';

describe('LearnArticle', () => {
  it('renders the missing-article fallback when id is not registered', () => {
    render(<LearnArticle articleId="this-does-not-exist" />);
    expect(screen.getByText(/no article registered/i)).toBeInTheDocument();
  });
});

describe('Article registry coverage', () => {
  it('renders every registered article without crashing', () => {
    for (const art of ARTICLES) {
      const { unmount, container } = render(
        <LearnArticle articleId={art.id} />,
      );
      expect(container.querySelector('.learn-article')).toBeInTheDocument();
      expect(container.querySelector('h1')).toHaveTextContent(art.title);
      unmount();
    }
  });

  it('every article references only equations that exist in EQUATIONS', async () => {
    const { EQUATIONS } = await import('../../lib/equations');
    for (const art of ARTICLES) {
      for (const sec of art.sections) {
        if (sec.kind === 'equation') {
          expect(EQUATIONS[sec.equationId], `${art.id} → ${sec.equationId}`).toBeTruthy();
        }
        if (sec.kind === 'workedExample') {
          for (const step of sec.example.steps) {
            if (step.equationId) {
              expect(EQUATIONS[step.equationId], `${art.id} step → ${step.equationId}`).toBeTruthy();
            }
          }
        }
        if (sec.kind === 'trainingDynamics') {
          expect(EQUATIONS[sec.updateRuleId], `${art.id} updateRule → ${sec.updateRuleId}`).toBeTruthy();
        }
      }
    }
  });
});
