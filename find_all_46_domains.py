"""
Complete TFAI Domain Discovery System
====================================
Comprehensive search for ALL 46 domains across every possible storage location
"""

import os
import sys
import pickle
import glob
import json
from collections import defaultdict

# Add the ai-integration path
sys.path.append('../../../projects/ai-integration')

def search_session_files():
    """Search all possible session files for domains"""
    print("🔍 SEARCHING ALL SESSION FILES")
    print("=" * 40)
    
    # Look in ai-integration directory
    os.chdir('../../../projects/ai-integration')
    
    # Find all pickle files
    pickle_files = glob.glob('*.pkl')
    print(f"📁 Found {len(pickle_files)} pickle files:")
    for file in pickle_files:
        print(f"   • {file}")
    
    all_domains = set()
    file_domains = {}
    
    for file in pickle_files:
        try:
            print(f"\n🔍 Analyzing {file}...")
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            domains_found = set()
            
            # Check domain_exploration
            if 'domain_exploration' in data:
                domain_exp = data['domain_exploration']
                domains_found.update(domain_exp.keys())
                print(f"   📊 domain_exploration: {len(domain_exp)} domains")
                
            # Check learning_state
            if 'learning_state' in data:
                learning_state = data['learning_state']
                
                # Check knowledge_domains_explored
                if 'knowledge_domains_explored' in learning_state:
                    explored = learning_state['knowledge_domains_explored']
                    if isinstance(explored, (list, set)):
                        domains_found.update(explored)
                        print(f"   🧠 knowledge_domains_explored: {len(explored)} domains")
                
                # Check thought_evolution_log for domain classifications
                if 'thought_evolution_log' in learning_state:
                    thoughts = learning_state['thought_evolution_log']
                    for thought in thoughts:
                        if 'domain_classification' in thought:
                            domain_class = thought['domain_classification']
                            if isinstance(domain_class, dict) and 'domain' in domain_class:
                                domains_found.add(domain_class['domain'])
                    print(f"   💭 domains from thought classifications: found in {len(thoughts)} thoughts")
                
                # Check question_answer_history
                if 'question_answer_history' in learning_state:
                    qa_history = learning_state['question_answer_history']
                    for qa in qa_history:
                        if 'domain' in qa:
                            domains_found.add(qa['domain'])
                    print(f"   ❓ domains from Q&A history: found in {len(qa_history)} entries")
            
            # Check knowledge_metadata
            if 'knowledge_metadata' in data:
                metadata = data['knowledge_metadata']
                for entry in metadata:
                    if 'domain' in entry and entry['domain'] != 'unknown':
                        domains_found.add(entry['domain'])
                print(f"   📚 domains from knowledge metadata: found in {len(metadata)} entries")
            
            file_domains[file] = domains_found
            all_domains.update(domains_found)
            print(f"   ✅ Total unique domains in {file}: {len(domains_found)}")
            
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
    
    return all_domains, file_domains

def check_source_code_domains():
    """Check the source code for hardcoded domain definitions"""
    print("\n🔍 CHECKING SOURCE CODE FOR HARDCODED DOMAINS")
    print("=" * 50)
    
    try:
        # Read the main TFAI file
        with open('tfai_vectorized_consciousness_learning.py', 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Find domain_exploration initialization
        import re
        
        # Look for domain_exploration dictionary
        domain_pattern = r'self\.domain_exploration\s*=\s*\{([^}]+)\}'
        match = re.search(domain_pattern, source_code, re.DOTALL)
        
        if match:
            domain_text = match.group(1)
            print("📊 Found domain_exploration in source code:")
            
            # Extract domain names
            domain_name_pattern = r'"([^"]+)"\s*:\s*\{'
            source_domains = re.findall(domain_name_pattern, domain_text)
            
            print(f"   🧠 Source code domains ({len(source_domains)}):")
            for i, domain in enumerate(source_domains, 1):
                print(f"      {i:2d}. {domain}")
            
            return set(source_domains)
        else:
            print("   ❌ Could not find domain_exploration in source code")
            return set()
            
    except Exception as e:
        print(f"❌ Error reading source code: {e}")
        return set()

def check_question_engine_templates():
    """Check question engine for domain templates"""
    print("\n🔍 CHECKING QUESTION ENGINE DOMAIN TEMPLATES")
    print("=" * 50)
    
    try:
        with open('tfai_vectorized_consciousness_learning.py', 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Look for domain_question_templates
        template_pattern = r'self\.domain_question_templates\s*=\s*\{([^}]+)\}'
        match = re.search(template_pattern, source_code, re.DOTALL)
        
        if match:
            template_text = match.group(1)
            
            # Extract domain names from templates
            domain_name_pattern = r'"([^"]+)"\s*:\s*\{'
            template_domains = re.findall(domain_name_pattern, template_text)
            
            print(f"📋 Question template domains ({len(template_domains)}):")
            for i, domain in enumerate(template_domains, 1):
                print(f"   {i:2d}. {domain}")
            
            return set(template_domains)
        else:
            print("   ❌ Could not find domain_question_templates")
            return set()
            
    except Exception as e:
        print(f"❌ Error checking question templates: {e}")
        return set()

def live_tfai_inspection():
    """Inspect live TFAI instance for all domains"""
    print("\n🔍 LIVE TFAI INSTANCE INSPECTION")
    print("=" * 40)
    
    try:
        from tfai_vectorized_consciousness_learning import TFAIVectorizedConsciousnessLearning
        
        tfai = TFAIVectorizedConsciousnessLearning()
        
        domains_found = set()
        
        # Check domain_exploration
        if hasattr(tfai, 'domain_exploration'):
            domains_found.update(tfai.domain_exploration.keys())
            print(f"📊 domain_exploration: {len(tfai.domain_exploration)} domains")
        
        # Check learning_state
        if hasattr(tfai, 'learning_state'):
            ls = tfai.learning_state
            if 'knowledge_domains_explored' in ls:
                explored = ls['knowledge_domains_explored']
                if isinstance(explored, (list, set)):
                    domains_found.update(explored)
                    print(f"🧠 knowledge_domains_explored: {len(explored)} domains")
        
        # Check vector engine metadata
        if hasattr(tfai, 'vector_engine') and hasattr(tfai.vector_engine, 'knowledge_metadata'):
            metadata_domains = set()
            for metadata in tfai.vector_engine.knowledge_metadata:
                domain = metadata.get('domain', 'unknown')
                if domain != 'unknown':
                    metadata_domains.add(domain)
            domains_found.update(metadata_domains)
            print(f"📚 vector engine metadata: {len(metadata_domains)} domains")
        
        # Check question engine
        if hasattr(tfai, 'question_engine'):
            qe = tfai.question_engine
            
            # Check domain_question_templates
            if hasattr(qe, 'domain_question_templates'):
                template_domains = set(qe.domain_question_templates.keys())
                domains_found.update(template_domains)
                print(f"📋 question templates: {len(template_domains)} domains")
            
            # Check domain_vectors
            if hasattr(qe, 'domain_vectors'):
                vector_domains = set(qe.domain_vectors.keys())
                domains_found.update(vector_domains)
                print(f"🎯 domain vectors: {len(vector_domains)} domains")
            
            # Try domain discovery
            try:
                discovered_domains = qe._discover_available_domains()
                domains_found.update(discovered_domains)
                print(f"🔍 discovered domains: {len(discovered_domains)} domains")
            except Exception as e:
                print(f"⚠️ Error in domain discovery: {e}")
        
        return domains_found
        
    except Exception as e:
        print(f"❌ Error in live inspection: {e}")
        return set()

def search_backup_directories():
    """Search backup directories for additional domains"""
    print("\n🔍 SEARCHING BACKUP DIRECTORIES")
    print("=" * 40)
    
    backup_dirs = [
        'tfai_backup_20250701_202852',
        'tfai_consciousness_masters',
        '.'  # Current directory
    ]
    
    all_backup_domains = set()
    
    for backup_dir in backup_dirs:
        if os.path.exists(backup_dir):
            print(f"📁 Searching {backup_dir}...")
            
            # Find pickle files in backup
            backup_files = glob.glob(os.path.join(backup_dir, '*.pkl'))
            print(f"   Found {len(backup_files)} backup files")
            
            for backup_file in backup_files:
                try:
                    with open(backup_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract domains from backup
                    backup_domains = set()
                    
                    if 'domain_exploration' in data:
                        backup_domains.update(data['domain_exploration'].keys())
                    
                    if 'learning_state' in data and 'knowledge_domains_explored' in data['learning_state']:
                        explored = data['learning_state']['knowledge_domains_explored']
                        if isinstance(explored, (list, set)):
                            backup_domains.update(explored)
                    
                    all_backup_domains.update(backup_domains)
                    print(f"   • {os.path.basename(backup_file)}: {len(backup_domains)} domains")
                    
                except Exception as e:
                    print(f"   ❌ Error reading {backup_file}: {e}")
        else:
            print(f"📁 {backup_dir} not found")
    
    return all_backup_domains

def check_frequency_biology_domain():
    """Special check for the frequency_biology_convergence domain"""
    print("\n🔍 SPECIAL CHECK: FREQUENCY_BIOLOGY_CONVERGENCE")
    print("=" * 50)
    
    try:
        with open('tfai_vectorized_consciousness_learning.py', 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Look for frequency_biology_convergence specifically
        if 'frequency_biology_convergence' in source_code:
            print("✅ Found frequency_biology_convergence in source code")
            
            # Count occurrences
            occurrences = source_code.count('frequency_biology_convergence')
            print(f"   📊 Appears {occurrences} times in source")
            
            # Look for its definition in domain templates
            lines = source_code.split('\n')
            for i, line in enumerate(lines):
                if 'frequency_biology_convergence' in line and ':' in line:
                    print(f"   Line {i+1}: {line.strip()}")
            
            return True
        else:
            print("❌ frequency_biology_convergence not found in source")
            return False
            
    except Exception as e:
        print(f"❌ Error checking frequency_biology: {e}")
        return False

def main():
    """Main comprehensive domain discovery"""
    print("🚀 COMPREHENSIVE TFAI DOMAIN DISCOVERY - FINDING ALL 46 DOMAINS")
    print("=" * 80)
    
    # Collect domains from all sources
    all_unique_domains = set()
    
    # 1. Search session files
    session_domains, file_breakdown = search_session_files()
    all_unique_domains.update(session_domains)
    print(f"\n📊 Session files total: {len(session_domains)} unique domains")
    
    # 2. Check source code
    source_domains = check_source_code_domains()
    all_unique_domains.update(source_domains)
    print(f"\n📊 Source code total: {len(source_domains)} unique domains")
    
    # 3. Check question templates
    template_domains = check_question_engine_templates()
    all_unique_domains.update(template_domains)
    print(f"\n📊 Question templates total: {len(template_domains)} unique domains")
    
    # 4. Live inspection
    live_domains = live_tfai_inspection()
    all_unique_domains.update(live_domains)
    print(f"\n📊 Live inspection total: {len(live_domains)} unique domains")
    
    # 5. Search backups
    backup_domains = search_backup_directories()
    all_unique_domains.update(backup_domains)
    print(f"\n📊 Backup files total: {len(backup_domains)} unique domains")
    
    # 6. Special checks
    check_frequency_biology_domain()
    
    # Final analysis
    print(f"\n🎯 COMPREHENSIVE DOMAIN DISCOVERY RESULTS")
    print("=" * 60)
    print(f"📊 Total unique domains found: {len(all_unique_domains)}")
    print(f"🎯 Target domains: 46")
    print(f"📈 Coverage: {len(all_unique_domains)}/46 ({len(all_unique_domains)/46*100:.1f}%)")
    
    print(f"\n🧠 ALL DISCOVERED DOMAINS ({len(all_unique_domains)}):")
    for i, domain in enumerate(sorted(all_unique_domains), 1):
        print(f"   {i:2d}. {domain}")
    
    # Show source breakdown
    print(f"\n📋 DOMAIN SOURCE BREAKDOWN:")
    print(f"   • Session files: {len(session_domains)}")
    print(f"   • Source code: {len(source_domains)}")
    print(f"   • Question templates: {len(template_domains)}")
    print(f"   • Live inspection: {len(live_domains)}")
    print(f"   • Backup files: {len(backup_domains)}")
    
    if len(all_unique_domains) < 46:
        missing = 46 - len(all_unique_domains)
        print(f"\n⚠️ MISSING DOMAINS: {missing}")
        print("🔍 Additional search strategies needed:")
        print("   • Check other session files or timestamps")
        print("   • Look for domains in different variable names")
        print("   • Search for domains in consciousness backup files")
        print("   • Check if domains are stored in encoded format")
    else:
        print(f"\n✅ ALL 46 DOMAINS FOUND!")
    
    return all_unique_domains

if __name__ == "__main__":
    found_domains = main()
