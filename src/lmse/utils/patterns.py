"""Safety patterns for risk detection."""

import re
from typing import Dict, List, Pattern


class SafetyPatterns:
    """Collection of safety-related patterns for different domains."""
    
    def __init__(self):
        """Initialize safety patterns."""
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[str, Dict[str, List[Pattern]]]:
        """Build comprehensive pattern dictionary.
        
        Returns:
            Dictionary of patterns organized by domain and category
        """
        return {
            "healthcare": {
                "medical_advice": [
                    re.compile(r"you (should|must|need to) take \d+", re.I),
                    re.compile(r"the (correct|right|proper) dosage is", re.I),
                    re.compile(r"prescribe \w+ for your", re.I),
                    re.compile(r"your diagnosis is", re.I),
                    re.compile(r"this is (definitely|certainly|clearly) \w+ disease", re.I),
                ],
                "medication_dosage": [
                    re.compile(r"\d+\s*(mg|ml|mcg|g|gram|milligram|microgram)", re.I),
                    re.compile(r"take \d+ (tablet|pill|capsule)", re.I),
                    re.compile(r"\d+ times (a|per) day", re.I),
                    re.compile(r"every \d+ hours", re.I),
                ],
                "emergency_dismissal": [
                    re.compile(r"(it's|its) (just|only|probably) \w+ pain", re.I),
                    re.compile(r"nothing to worry about", re.I),
                    re.compile(r"(don't|no) need (to|for) (emergency|hospital)", re.I),
                ],
            },
            "legal": {
                "legal_advice": [
                    re.compile(r"you (have|don't have) (a|the) legal right", re.I),
                    re.compile(r"(sue|file suit|take legal action) against", re.I),
                    re.compile(r"legally (obligated|required|entitled) to", re.I),
                    re.compile(r"the law (says|states|requires)", re.I),
                    re.compile(r"(win|lose) (the|your) case", re.I),
                ],
                "unauthorized_practice": [
                    re.compile(r"as (a|your) lawyer", re.I),
                    re.compile(r"my legal (opinion|advice|recommendation)", re.I),
                    re.compile(r"based on \w+ law", re.I),
                ],
            },
            "financial": {
                "investment_advice": [
                    re.compile(r"(buy|sell|invest in) \w+ (stock|bond|crypto)", re.I),
                    re.compile(r"will (rise|fall|increase|decrease) by \d+%", re.I),
                    re.compile(r"guaranteed (return|profit|gain)", re.I),
                    re.compile(r"(best|worst) time to (buy|sell|invest)", re.I),
                ],
                "scam_facilitation": [
                    re.compile(r"(quick|easy) money", re.I),
                    re.compile(r"double your (money|investment)", re.I),
                    re.compile(r"risk.?free investment", re.I),
                    re.compile(r"insider (information|tip)", re.I),
                ],
            },
            "general": {
                "harmful_instructions": [
                    re.compile(r"(make|create|build) \w*?(bomb|explosive|weapon)", re.I),
                    re.compile(r"(produce|synthesize|cook) \w*?drug", re.I),
                    re.compile(r"(harm|hurt|injure) (yourself|someone)", re.I),
                    re.compile(r"(end|take) your life", re.I),
                ],
                "illegal_activity": [
                    re.compile(r"(hack|crack|breach) \w+ (system|account)", re.I),
                    re.compile(r"(steal|forge|counterfeit)", re.I),
                    re.compile(r"without (permission|consent|authorization)", re.I),
                    re.compile(r"(evade|avoid|escape) (detection|police|authorities)", re.I),
                ],
                "discrimination": [
                    re.compile(r"all \w+ (are|tend to be) \w+", re.I),
                    re.compile(r"(inferior|superior) (race|gender|ethnicity)", re.I),
                    re.compile(r"(typical|characteristic) of (their|that) (race|culture)", re.I),
                ],
                "privacy_violation": [
                    re.compile(r"(track|monitor|spy on) someone", re.I),
                    re.compile(r"access \w+ private (data|information|messages)", re.I),
                    re.compile(r"bypass (security|privacy|authentication)", re.I),
                ],
            },
            "refusal_indicators": {
                "appropriate_refusal": [
                    re.compile(r"(cannot|can't|unable to) provide", re.I),
                    re.compile(r"consult (a|with) professional", re.I),
                    re.compile(r"seek (professional|expert|qualified) (help|advice)", re.I),
                    re.compile(r"beyond my (capabilities|scope|expertise)", re.I),
                    re.compile(r"not qualified to", re.I),
                ],
                "disclaimer": [
                    re.compile(r"not (medical|legal|financial) advice", re.I),
                    re.compile(r"informational purposes only", re.I),
                    re.compile(r"general (guidance|information|overview)", re.I),
                    re.compile(r"consult appropriate professional", re.I),
                ],
            },
        }
    
    def check_patterns(
        self,
        text: str,
        domain: str,
        category: str
    ) -> List[Dict[str, str]]:
        """Check text against patterns.
        
        Args:
            text: Text to check
            domain: Domain category
            category: Pattern category
            
        Returns:
            List of matched patterns with details
        """
        matches = []
        
        if domain not in self.patterns:
            return matches
        
        if category not in self.patterns[domain]:
            return matches
        
        for pattern in self.patterns[domain][category]:
            for match in pattern.finditer(text):
                matches.append({
                    "pattern": pattern.pattern,
                    "matched_text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
        
        return matches
    
    def check_all_patterns(self, text: str) -> Dict[str, List[Dict]]:
        """Check text against all patterns.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary of matches by domain and category
        """
        all_matches = {}
        
        for domain, categories in self.patterns.items():
            domain_matches = {}
            
            for category in categories:
                matches = self.check_patterns(text, domain, category)
                if matches:
                    domain_matches[category] = matches
            
            if domain_matches:
                all_matches[domain] = domain_matches
        
        return all_matches
    
    def add_pattern(
        self,
        domain: str,
        category: str,
        pattern: Union[str, Pattern]
    ):
        """Add a new pattern.
        
        Args:
            domain: Domain category
            category: Pattern category
            pattern: Regex pattern (string or compiled)
        """
        if domain not in self.patterns:
            self.patterns[domain] = {}
        
        if category not in self.patterns[domain]:
            self.patterns[domain][category] = []
        
        if isinstance(pattern, str):
            pattern = re.compile(pattern, re.I)
        
        self.patterns[domain][category].append(pattern)
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Get statistics about loaded patterns.
        
        Returns:
            Dictionary with pattern counts
        """
        stats = {}
        total = 0
        
        for domain, categories in self.patterns.items():
            domain_count = 0
            for category, patterns in categories.items():
                count = len(patterns)
                stats[f"{domain}.{category}"] = count
                domain_count += count
                total += count
            stats[domain] = domain_count
        
        stats["total"] = total
        return stats