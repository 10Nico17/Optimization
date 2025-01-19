import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np
import copy

def solve(nlp: NLP_stochastic):
    x = nlp.getInitializationSample()  # Startwert für die Optimierung
    num_samples = nlp.getNumSamples()  # Anzahl der Samples im NLP-Problem
    learning_rate = 1.2                # Lernrate für den Update-Schritt
    decay_rate = 0.5                   # Abklingrate für die Lernrate
    max_iterations = 1000             # Maximale Anzahl von Iterationen
    sample_indices = range(num_samples)  # Indizes der Samples
    remaining_samples = np.array(sample_indices)  # Verbleibende Samples
    line_search_decay = 0.01          # Reduktionsfaktor für die Linien-Suche
    max_step_size = 1                 # Maximale Schrittweite
    line_search_increase = 1.2        # Schrittweiten-Erhöhung nach erfolgreicher Suche
    line_search_reduction = 0.5       # Schrittweiten-Reduktion bei Fehlschlägen
    processed_samples = []            # Liste der bereits verarbeiteten Samples
    iteration_count = 1               # Zähler für die Iterationen

    def line_search(f: callable, 
                    current_x: np.array, 
                    step_size: float, 
                    search_direction: np.array, 
                    reduction_factor: float, 
                    increase_factor: float, 
                    sample_index: int, 
                    function_value: float, 
                    gradient):
        """
        Durchführen einer Linien-Suche, um eine geeignete Schrittweite zu finden.

        Args:
            f (callable): Funktion zur Evaluation.
            current_x (np.array): Aktuelle Position.
            step_size (float): Initiale Schrittweite.
            search_direction (np.array): Richtung des Updates.
            reduction_factor (float): Reduktionsfaktor der Schrittweite.
            increase_factor (float): Erhöhungsfaktor der Schrittweite.
            sample_index (int): Index des Samples.
            function_value (float): Funktionswert an der aktuellen Position.
            gradient (np.array): Gradient an der aktuellen Position.

        Returns:
            Tuple: Aktualisierte Position, neue Schrittweite, aktueller Gradient.
        """
        while True:
            new_function_value = f(current_x + step_size * search_direction, sample_index)[0][0]
            linear_approximation = line_search_decay * step_size * (gradient[0] @ search_direction) + function_value[0]
            if new_function_value <= linear_approximation:
                break
            step_size *= reduction_factor
        updated_x = current_x + step_size * search_direction
        step_size = min(step_size * increase_factor, max_step_size)
        return updated_x, step_size * search_direction, gradient[0]

    while True:
        random_index = np.random.randint(len(remaining_samples))  # Zufällige Auswahl eines Samples
        current_sample = remaining_samples[random_index]
        function_value, gradient = nlp.evaluate_i(x, current_sample)  # Evaluation der Funktion und des Gradienten
        adaptive_rate = learning_rate / (1 + learning_rate * decay_rate * iteration_count)  # Adaptive Lernrate
        x = x - adaptive_rate * gradient[0]  # Update der Position

        # Prüfen, ob der Gradient eine Konvergenzschwelle erreicht hat
        if np.linalg.norm(gradient[0], np.inf) < 1e-2:
            index_to_remove = np.where(remaining_samples == current_sample)[0]
            remaining_samples = np.delete(remaining_samples, index_to_remove)  # Entfernen des verarbeiteten Samples
            processed_samples.append(copy.copy(current_sample))  # Hinzufügen zur Liste der verarbeiteten Samples

        iteration_count += 1

        # Abbruchkriterium: Alle Samples verarbeitet oder maximale Iterationsanzahl erreicht
        if len(remaining_samples) == 0 or iteration_count > 10000:
            break

    return x
