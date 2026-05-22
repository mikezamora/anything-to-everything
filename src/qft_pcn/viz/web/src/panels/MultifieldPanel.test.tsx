import { render } from '@testing-library/react';
import { it } from 'vitest';
import { MultifieldPanel } from './MultifieldPanel';
import { multifieldFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<MultifieldPanel frame={multifieldFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MultifieldPanel frame={emptyFrame} />);
});
