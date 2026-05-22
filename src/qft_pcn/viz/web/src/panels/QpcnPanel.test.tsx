import { render } from '@testing-library/react';
import { it } from 'vitest';
import { QpcnPanel } from './QpcnPanel';
import { qpcnFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<QpcnPanel frame={qpcnFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<QpcnPanel frame={emptyFrame} />);
});
