package dcs;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import static fig.basic.LogInfo.*;

/**
 * Contains a function and a current point.
 * Convention: if we want value() and gradient() at the current point,
 * probably wise to call gradient() first.
 */
interface FunctionState {
  // The current point
  public double[] point();

  // Function value at the current point
  public double value();

  // Gradient at the current point
  // The gradient is owned by the FunctionState,
  // so the caller must make a copy if he wants
  // Contract: the array gradient is overwritten only gradient is called again
  // (with an invalidated point).
  public double[] gradient();

  // Need to call this whenever the current point is changed
  // so next time we know to recompute the value and gradient
  public void invalidate();
}

interface Maximizer {
  // Return if we've already reached the maximum.
  // Just take one gradient step somehow.
  public abstract boolean takeStep(FunctionState func);
  public void logStats();
}

class GradientMaximizer implements Maximizer {
  BacktrackingLineSearch lineSearch;

  public GradientMaximizer(BacktrackingLineSearch.Options btopts) {
    this.lineSearch = new BacktrackingLineSearch(btopts);
  }

  public boolean takeStep(FunctionState func) {
    double[] gradient = func.gradient();
    logs("value = %s", Fmt.D(func.value()));
    logs("gradientNormSquared = %s", Fmt.D(NumUtils.l2NormSquared(gradient)));
    return lineSearch.maximize(func, getDirection(func));
  }

  // Can override
  protected double[] getDirection(FunctionState func) {
    return func.gradient();
  }

  public void logStats() {
    logss("numReductions: " + lineSearch.numReductionsFig);
  }
}

class LBFGSMaximizer extends GradientMaximizer {
  public static class Options {
    @Option public int historySize = 5;
  }

  Options opts;
  BacktrackingLineSearch lineSearch;

  // Implicitly store the approximation of the inverse Hessian
  LinkedList<double[]> sHistory; // History of point differences
  LinkedList<double[]> yHistory; // History of gradient differences
  double[] lastx;
  double[] lastg;

  public LBFGSMaximizer(BacktrackingLineSearch.Options btopts, Options opts) {
    super(btopts);
    this.opts = opts;
    this.sHistory = new LinkedList();
    this.yHistory = new LinkedList();
  }

  // Add src1-src2 to history (kick out oldest if necessary to maintain history size)
  public void store(LinkedList<double[]> history, double[] src1, double[] src2) {
    double[] dest;
    if(history.size() < opts.historySize)
      dest = new double[src1.length];
    else
      dest = history.removeFirst(); // Reuse old vectors
    for(int i = 0; i < dest.length; i++)
      dest[i] = src1[i] - src2[i];
    history.addLast(dest);
  }

  public double[] precondition(double[] g) {
    double[] p = new double[g.length]; ListUtils.incr(p, -1, g);
    int n = yHistory.size(); // Don't include last point
    double[] sy = new double[n]; // s^T y
    double[] alpha = new double[n];

    // Backward pass
    for(int k = n-1; k >= 0; k--) {
      sy[k] = ListUtils.dot(sHistory.get(k), yHistory.get(k));
      if (sy[k] <= 0) {
        errors("Computing precondition failed: sy[k=%s] = %s is not positive, clearing the history", k, sy[k]);
        yHistory.clear();
        ListUtils.set(p, 0);
        ListUtils.incr(p, -1, g);
        return p;
      }
      alpha[k] = ListUtils.dot(sHistory.get(k), p) / sy[k];
      ListUtils.incr(p, -alpha[k], yHistory.get(k));
    }
    if(n > 0)
      ListUtils.multMut(p,
          sy[n-1] / ListUtils.dot(yHistory.get(n-1), yHistory.get(n-1)));
    // Forward pass
    for(int k = 0; k < n; k++) {
      double beta = ListUtils.dot(yHistory.get(k), p) / sy[k];
      ListUtils.incr(p, (alpha[k] - beta), sHistory.get(k));
    }

    //dbg("g: " + Fmt.D(g));
    //dbg("p: " + Fmt.D(p));
    return p;
  }

  public double[] getDirection(FunctionState func) {
    double[] x = func.point();
    double[] g = func.gradient();
    ListUtils.multMut(g, -1); // Pretend we're minimizing the function
    double[] p = precondition(g); // Use current approximation of inverse Hessian
    if(lastx == null) {
      // First point - just store, inverse Hessian approximation is identity
      lastx = new double[x.length];
      lastg = new double[g.length];
    }
    else {
      store(sHistory, x, lastx); // s_k = x_{k+1} - x_k
      store(yHistory, g, lastg); // y_k = g_{k+1} - g_k
    }
    ListUtils.set(lastx, x);
    ListUtils.set(lastg, g);
    return p;
  }

  public static void main(String[] args) {
    BacktrackingLineSearch.Options btopts = new BacktrackingLineSearch.Options();
    LBFGSMaximizer.Options lopts = new LBFGSMaximizer.Options();
    Execution.init(args, "lsearch", btopts, "lbfgs", lopts);
    Maximizer maximizer = new GradientMaximizer(btopts);
    maximizer = new LBFGSMaximizer(btopts, lopts);
    FunctionState state = new FunctionState() {
      private double[] x = new double[2];
      public double[] point() { return x; }
      private double truePoint(int i) { return i+3; }
      private double scale(int i) { return (i+1)*5; }
      public double[] gradient() {
        double[] dx = new double[x.length];
        for(int i = 0; i < x.length; i++)
          dx[i] = -scale(i) * (x[i] - truePoint(i));
        return dx;
      }
      public double value() {
        double y = 0;
        for(int i = 0; i < x.length; i++)
          y -= 0.5 * scale(i) * (x[i] - truePoint(i))*(x[i] - truePoint(i));
        return y;
      }
      public void invalidate() { }
    };
    for(int iter = 0; ; iter++) {
      logs("iter %d: x = %s, y = %s, dx = %s", iter,
        Fmt.D(state.point()),
        Fmt.D(state.value()),
        Fmt.D(state.gradient()));
      if(maximizer.takeStep(state)) break;
    }
    Execution.finish();
  }
}

class BacktrackingLineSearch {
  public static class Options {
    @Option public double stepSizeDecrFactor = 0.9;
    @Option public double stepSizeIncrFactor = 1.3;
    @Option public int maxTries = 100;
    @Option public double tolerance = 1e-5;
    @Option public double initStepSize = 1;
  }

  // Current step size: maintain this across optimization iterations
  double stepSize;
  FullStatFig numReductionsFig = new FullStatFig();
  Options opts;

  public BacktrackingLineSearch(Options opts) {
    this.opts = opts;
    this.stepSize = opts.initStepSize;
  }
    
  // Very crude search: decrease step size until get improvement
  // Return whether can't decrease it anymore
  public boolean maximize(FunctionState func, double[] dx) {
    // Heuristic for choosing the initial step size
    //if(stepSize == 0) stepSize = 30.0 / NumUtils.l2NormSquared(dx);
    // 328 0.089
    // 492 0.058
    // 113 0.729 (win=0,edge=false)

    track("BacktrackingLineSearch(stepSize=%s)", Fmt.D(stepSize));
    double[] x = func.point();
    double y0 = func.value();
    double oldStepSize = 0;
    double lasty = Double.NEGATIVE_INFINITY;
    //dbg(Fmt.D(x));
    int t;
    for(t = 0; ; t++) {
      // Keep on decreasing the step size until things start getting worse
      ListUtils.incr(x, stepSize-oldStepSize, dx);
      //dbgs("x = %s", Fmt.D(x));
      //dbgs("dx = %s", Fmt.D(dx));
      func.invalidate();
      double y = func.value();
      logs("value = %s (stepSize = %s)", Fmt.D(y), Fmt.D(stepSize));

      // y is worse than lasty, so lasty is the best we have
      // (or we've reached the maximum number of reductions)
      if(y < lasty || t+1 == opts.maxTries) {
        numReductionsFig.add(t+1);
        if(lasty > y0 + opts.tolerance) { // lasty is an improvement overall (by a non-negligible amount)
          // Roll back to lasty
          ListUtils.incr(x, oldStepSize-stepSize, dx);
          stepSize = oldStepSize * opts.stepSizeIncrFactor; // Increase step size for next time
          logs("Improvement of %s on reduction %d", Fmt.D(lasty-y0), t);
          end_track();
          return false;
        }
        else {
          // For convex function, impossible for things to get better and then worse
          // Roll back to y0: unable to reduce
          ListUtils.incr(x, -stepSize, dx);
          logs("No improvement after %d reductions, stopping", t);
          end_track();
          return true;
        }
      }

      // Cut the stepSize
      oldStepSize = stepSize;
      stepSize *= opts.stepSizeDecrFactor;
      lasty = y;
    }
  }
}
