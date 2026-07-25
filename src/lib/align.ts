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

/**
 * Symmetric (neutral) alignment: rather than mapping A onto B, return the two
 * rotations that carry each set into the SVD's shared middle frame. Applying
 * `left` to the A-space and `right` to the B-space makes the anchor sets'
 * cross-covariance diagonal (`Aᵀ·B = U·Σ·Vᵀ` ⇒ `(A·U)ᵀ(B·V) = Σ`), i.e. aligned
 * axis-by-axis with neither space privileged.
 */
export function symmetricRotations(
  A: number[][],
  B: number[][],
): { left: Matrix; right: Matrix } {
  const svd = new SingularValueDecomposition(
    new Matrix(A).transpose().mmul(new Matrix(B)),
  )
  return { left: svd.leftSingularVectors, right: svd.rightSingularVectors }
}

/** Project rows onto their top-2 principal components (mean-centered): `N×d → N×2`. */
export function pca2d(vectors: number[][]): number[][] {
  if (!vectors.length) return []
  const M = new Matrix(vectors)
  const rows = M.rows
  const cols = M.columns
  const mean = new Array<number>(cols).fill(0)
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) mean[j] += M.get(i, j)
  for (let j = 0; j < cols; j++) mean[j] /= rows
  const centered = M.clone()
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      centered.set(i, j, centered.get(i, j) - mean[j])
  const V = new SingularValueDecomposition(centered).rightSingularVectors
  const top2 = V.subMatrix(0, V.rows - 1, 0, Math.min(1, V.columns - 1))
  return centered.mmul(top2).to2DArray()
}
