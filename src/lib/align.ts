import { Matrix, SingularValueDecomposition } from 'ml-matrix'

/**
 * Orthogonal Procrustes alignment of two point sets in a shared dimensionality.
 *
 * Given anchor rows `A` (source) and `B` (target), both `k × d`, returns the
 * orthogonal `d × d` matrix `R` minimising ‖A·R − B‖ — i.e. the rotation/reflection
 * that best maps the source space onto the target space (Schönemann 1966). With
 * `k < d` anchors the map is only determined on the anchors' span; the rest is an
 * arbitrary rotation (few anchors ⇒ unreliable alignment — surfaced, not hidden).
 *
 * `R = U·Vᵀ` where `U·Σ·Vᵀ = SVD(Aᵀ·B)`.
 */
export function procrustes(A: number[][], B: number[][]): Matrix {
  const M = new Matrix(A).transpose().mmul(new Matrix(B)) // d × d
  const svd = new SingularValueDecomposition(M)
  return svd.leftSingularVectors.mmul(svd.rightSingularVectors.transpose())
}

/** Map each row vector through `R` (returns `vectors · R`). */
export function applyRotation(vectors: number[][], R: Matrix): number[][] {
  return new Matrix(vectors).mmul(R).to2DArray()
}
