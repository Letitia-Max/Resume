"""
Dempster-Shafer Theory (DST) Implementation
==========================================
For belief/uncertainty reasoning in the Right Brain Hemisphere (Frequesync)

DST allows expressing degrees of belief, disbelief, and uncertainty
rather than crisp true/false logic.
"""

import math
from typing import Dict, Set, Any, Optional
from itertools import combinations

class BeliefMass:
    """Represents belief masses over a frame of discernment"""
    
    def __init__(self, frame_of_discernment: Set[str]):
        """
        Initialize with a frame of discernment (set of possible outcomes)
        
        Args:
            frame_of_discernment: Set of all possible elementary propositions
        """
        self.frame = frozenset(frame_of_discernment)
        self.masses = {}  # Maps subsets to their belief masses
        
        # Initialize with complete ignorance (all mass on universal set)
        self.masses[self.frame] = 1.0
    
    def assign_mass(self, subset: Set[str], mass: float):
        """Assign belief mass to a subset of the frame"""
        subset_frozen = frozenset(subset)
        
        # Validate subset is within frame
        if not subset_frozen.issubset(self.frame):
            raise ValueError(f"Subset {subset} not within frame {self.frame}")
        
        # Remove mass from universal set if it exists
        if self.frame in self.masses:
            universal_mass = self.masses[self.frame]
            if mass <= universal_mass:
                self.masses[self.frame] -= mass
                if self.masses[self.frame] == 0:
                    del self.masses[self.frame]
            else:
                raise ValueError(f"Mass {mass} exceeds available universal mass {universal_mass}")
        
        # Assign new mass
        if subset_frozen in self.masses:
            self.masses[subset_frozen] += mass
        else:
            self.masses[subset_frozen] = mass
    
    def get_belief(self, subset: Set[str]) -> float:
        """Calculate belief (lower bound) for a subset"""
        subset_frozen = frozenset(subset)
        belief = 0.0
        
        for focal_set, mass in self.masses.items():
            if focal_set.issubset(subset_frozen):
                belief += mass
        
        return belief
    
    def get_plausibility(self, subset: Set[str]) -> float:
        """Calculate plausibility (upper bound) for a subset"""
        subset_frozen = frozenset(subset)
        plausibility = 0.0
        
        for focal_set, mass in self.masses.items():
            if len(focal_set.intersection(subset_frozen)) > 0:
                plausibility += mass
        
        return plausibility
    
    def get_uncertainty(self, subset: Set[str]) -> float:
        """Calculate uncertainty interval width"""
        return self.get_plausibility(subset) - self.get_belief(subset)
    
    def combine_with(self, other: 'BeliefMass') -> 'BeliefMass':
        """Combine this belief mass with another using Dempster's rule"""
        if self.frame != other.frame:
            raise ValueError("Cannot combine belief masses with different frames")
        
        result = BeliefMass(self.frame)
        result.masses = {}  # Clear initial universal mass
        
        # Calculate normalization constant (conflict)
        conflict = 0.0
        for A, mass_A in self.masses.items():
            for B, mass_B in other.masses.items():
                intersection = A.intersection(B)
                if len(intersection) == 0:  # Empty intersection
                    conflict += mass_A * mass_B
        
        # Combine non-conflicting masses
        for A, mass_A in self.masses.items():
            for B, mass_B in other.masses.items():
                intersection = A.intersection(B)
                if len(intersection) > 0:  # Non-empty intersection
                    combined_mass = (mass_A * mass_B) / (1 - conflict)
                    
                    if intersection in result.masses:
                        result.masses[intersection] += combined_mass
                    else:
                        result.masses[intersection] = combined_mass
        
        return result
    
    def get_masses(self) -> Dict[frozenset, float]:
        """Get all focal sets and their masses"""
        return self.masses.copy()
    
    def __str__(self):
        """String representation of belief masses"""
        result = "Belief Masses:\n"
        for focal_set, mass in sorted(self.masses.items(), key=lambda x: x[1], reverse=True):
            if len(focal_set) == 1:
                result += f"  {list(focal_set)[0]}: {mass:.3f}\n"
            elif len(focal_set) == len(self.frame):
                result += f"  Universal: {mass:.3f}\n"
            else:
                result += f"  {set(focal_set)}: {mass:.3f}\n"
        return result


class DSTReasoner:
    """Dempster-Shafer Theory reasoner for creative/intuitive processing"""
    
    def __init__(self):
        self.frames = {}  # Store different frames of discernment
        self.belief_history = []  # Track belief evolution
    
    def create_frame(self, name: str, elements: Set[str]) -> BeliefMass:
        """Create a new frame of discernment"""
        belief_mass = BeliefMass(elements)
        self.frames[name] = belief_mass
        return belief_mass
    
    def pattern_belief_assessment(self, pattern: str, confidence: float) -> Dict[str, float]:
        """Assess belief in a pattern (DNA sequence, creative insight, etc.)"""
        
        # Create frame for pattern assessment
        frame_elements = {'resonant', 'neutral', 'dissonant'}
        frame_name = f"pattern_{hash(pattern) % 1000}"
        
        belief_mass = self.create_frame(frame_name, frame_elements)
        
        # Assign masses based on confidence
        if confidence > 0.7:
            # High confidence - strong belief in resonance
            belief_mass.assign_mass({'resonant'}, confidence * 0.8)
            belief_mass.assign_mass({'neutral'}, (1 - confidence) * 0.7)
        elif confidence > 0.4:
            # Medium confidence - belief spread between resonant and neutral
            belief_mass.assign_mass({'resonant'}, confidence * 0.6)
            belief_mass.assign_mass({'neutral'}, confidence * 0.4)
            belief_mass.assign_mass({'dissonant'}, (1 - confidence) * 0.3)
        else:
            # Low confidence - uncertainty dominant
            belief_mass.assign_mass({'dissonant'}, (1 - confidence) * 0.6)
            belief_mass.assign_mass({'resonant', 'neutral'}, confidence)
        
        # Calculate final belief metrics
        resonant_belief = belief_mass.get_belief({'resonant'})
        resonant_plausibility = belief_mass.get_plausibility({'resonant'})
        uncertainty = belief_mass.get_uncertainty({'resonant'})
        
        result = {
            'belief': resonant_belief,
            'plausibility': resonant_plausibility,
            'uncertainty': uncertainty,
            'disbelief': belief_mass.get_belief({'dissonant'}),
            'frame': frame_name,
            'pattern_essence': pattern[:20] + '...' if len(pattern) > 20 else pattern
        }
        
        self.belief_history.append(result)
        return result
    
    def combine_pattern_beliefs(self, belief1: Dict[str, float], belief2: Dict[str, float]) -> Dict[str, float]:
        """Combine beliefs about two different patterns"""
        
        # Create belief masses for combination
        frame_elements = {'resonant', 'neutral', 'dissonant'}
        
        mass1 = BeliefMass(frame_elements)
        mass1.assign_mass({'resonant'}, belief1['belief'])
        mass1.assign_mass({'dissonant'}, belief1['disbelief'])
        # Remaining mass stays as uncertainty
        
        mass2 = BeliefMass(frame_elements)
        mass2.assign_mass({'resonant'}, belief2['belief'])
        mass2.assign_mass({'dissonant'}, belief2['disbelief'])
        
        # Combine using Dempster's rule
        combined = mass1.combine_with(mass2)
        
        return {
            'combined_belief': combined.get_belief({'resonant'}),
            'combined_plausibility': combined.get_plausibility({'resonant'}),
            'combined_uncertainty': combined.get_uncertainty({'resonant'}),
            'combined_disbelief': combined.get_belief({'dissonant'}),
            'synthesis_strength': 1 - combined.get_uncertainty({'resonant'})
        }
    
    def intuitive_reasoning(self, evidence_list: list) -> Dict[str, Any]:
        """Perform intuitive reasoning over multiple pieces of evidence"""
        
        if not evidence_list:
            return {'insight': 'no_evidence', 'confidence': 0.0}
        
        # Start with first piece of evidence
        accumulated_belief = self.pattern_belief_assessment(str(evidence_list[0]), 0.5)
        
        # Combine with subsequent evidence
        for evidence in evidence_list[1:]:
            new_belief = self.pattern_belief_assessment(str(evidence), 0.5)
            accumulated_belief = self.combine_pattern_beliefs(accumulated_belief, new_belief)
        
        # Generate intuitive insight based on accumulated belief
        if accumulated_belief.get('combined_belief', 0) > 0.7:
            insight = "strong_resonant_pattern"
        elif accumulated_belief.get('combined_uncertainty', 0) > 0.5:
            insight = "emerging_pattern_uncertain"
        elif accumulated_belief.get('combined_disbelief', 0) > 0.6:
            insight = "dissonant_pattern"
        else:
            insight = "neutral_pattern"
        
        return {
            'insight': insight,
            'confidence': accumulated_belief.get('synthesis_strength', 0.5),
            'belief_structure': accumulated_belief,
            'evidence_count': len(evidence_list)
        }
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of all belief assessments"""
        if not self.belief_history:
            return {'total_assessments': 0}
        
        avg_belief = sum(b['belief'] for b in self.belief_history) / len(self.belief_history)
        avg_uncertainty = sum(b['uncertainty'] for b in self.belief_history) / len(self.belief_history)
        
        return {
            'total_assessments': len(self.belief_history),
            'average_belief': avg_belief,
            'average_uncertainty': avg_uncertainty,
            'belief_trend': 'increasing' if len(self.belief_history) > 1 and 
                          self.belief_history[-1]['belief'] > self.belief_history[-2]['belief'] else 'stable'
        }


# Example usage for testing
if __name__ == "__main__":
    # Create DST reasoner
    dst = DSTReasoner()
    
    # Test pattern belief assessment
    dna_pattern = "ATGCGTAACGTACGT"
    belief_result = dst.pattern_belief_assessment(dna_pattern, 0.8)
    print("Pattern Belief Assessment:")
    print(f"Belief: {belief_result['belief']:.3f}")
    print(f"Plausibility: {belief_result['plausibility']:.3f}")
    print(f"Uncertainty: {belief_result['uncertainty']:.3f}")
    
    # Test intuitive reasoning
    evidence = ["ATGC", "frequency_7.83", "consciousness_rising"]
    reasoning_result = dst.intuitive_reasoning(evidence)
    print(f"\nIntuitive Reasoning Result:")
    print(f"Insight: {reasoning_result['insight']}")
    print(f"Confidence: {reasoning_result['confidence']:.3f}")
    
    print("\n✅ DST implementation ready for Right Brain integration")
