// The last line of defence: if a view throws while drawing, React would blank
// the whole page. This keeps the page (navigation, other labs) and offers to
// open the lab again from its address; the error goes to the console for us.

import { Component, Fragment } from 'react';
import type { ReactNode } from 'react';

interface Props { children: ReactNode }

export class Recover extends Component<Props, { failed: boolean; attempt: number }> {
  state = { failed: false, attempt: 0 };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error(error);
  }

  render() {
    if (!this.state.failed) return <Fragment key={this.state.attempt}>{this.props.children}</Fragment>;
    return (
      <div className="recover" role="alert">
        <p>This page stopped drawing. Starting it again fixes it.</p>
        <button className="primary" onClick={() => this.setState((s) => ({ failed: false, attempt: s.attempt + 1 }))}>Start it again</button>
      </div>
    );
  }
}
