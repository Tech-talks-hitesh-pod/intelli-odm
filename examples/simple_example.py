#!/usr/bin/env python3
"""
Simple example of using the Intelli-ODM system.

This script demonstrates basic usage of the orchestrator agent
to analyze products and generate procurement recommendations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from shared_knowledge_base import SharedKnowledgeBase
from utils.llm_client import LLMClientFactory
from agents.orchestrator_agent import OrchestratorAgent

def main():
    print("🚀 Intelli-ODM Simple Example")
    print("=" * 40)
    
    # Setup
    try:
        settings = Settings()
        print(f"📊 Using LLM Provider: {settings.llm_provider}")
        
        # Create LLM client
        if settings.llm_provider == "ollama":
            llm_config = {
                "provider": "ollama",
                "base_url": settings.ollama_base_url,
                "model": settings.ollama_model
            }
        else:
            if not settings.openai_api_key:
                print("❌ OpenAI API key required. Please set OPENAI_API_KEY environment variable.")
                return
            llm_config = {
                "provider": "openai", 
                "api_key": settings.openai_api_key,
                "model": settings.openai_model
            }
        
        llm_client = LLMClientFactory.create_client(llm_config)
        print("✅ LLM client initialized")
        
        # Initialize knowledge base
        kb = SharedKnowledgeBase(persist_directory="data/chroma_db")
        print("✅ Knowledge base initialized")
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(llm_client, kb)
        print("✅ Orchestrator initialized")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return
    
    # Sample data
    products = [
        "White cotton t-shirt, crew neck, short sleeves, regular fit",
        "Blue denim jeans, slim fit, dark wash, button fly",
        "Black cocktail dress, sleeveless, knee-length, chiffon"
    ]
    
    inventory = {
        "product_1": 50,
        "product_2": 30, 
        "product_3": 10
    }
    
    print(f"\n📦 Analyzing {len(products)} products...")
    
    # Execute workflow
    try:
        results = orchestrator.run_complete_workflow(
            product_descriptions=products,
            inventory_data=inventory
        )
        
        if results['success']:
            print("\n✅ Analysis completed successfully!")
            print(f"📊 Confidence Score: {results['confidence_score']:.2f}")
            print(f"📈 Success Rate: {results['metrics']['success_rate']:.1%}")
            
            # Show recommendations
            recommendations = results['recommendations']['procurement']
            if recommendations:
                print(f"\n🛒 Procurement Recommendations:")
                for rec in recommendations:
                    print(f"  • {rec['product_id']}: {rec['quantity']} units (${rec['estimated_cost']:,.0f})")
                    print(f"    Priority: {rec['priority']}, Timeline: {rec['timeline']}")
            
            # Show insights
            print(f"\n🔍 Key Insights:")
            for insight in results['key_insights']:
                print(f"  • {insight}")
                
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Workflow failed: {e}")

if __name__ == "__main__":
    main()