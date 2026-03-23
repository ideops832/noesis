//! Core hypervector trait and common types.

/// Trait comune per tutti i tipi di ipervettore.
pub trait HyperVector {
    /// Dimensionalità logica dell'ipervettore.
    fn dim(&self) -> usize;

    /// Similarità con un altro ipervettore dello stesso tipo.
    fn similarity(&self, other: &Self) -> f32;
}
