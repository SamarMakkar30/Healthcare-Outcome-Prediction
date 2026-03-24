"""
Healthcare Prediction System - Production Entry Point
Run the FastAPI application with production settings.
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare Prediction System"
    )
    parser.add_argument(
        '--mode', 
        choices=['api', 'train', 'test'],
        default='api',
        help='Run mode: api (default), train, or test'
    )
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='API host (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='API port (default: 8000)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1,
        help='Number of workers (default: 1)'
    )
    parser.add_argument(
        '--reload', 
        action='store_true',
        help='Enable auto-reload (development only)'
    )
    parser.add_argument(
        '--disease',
        choices=['diabetes', 'heart_disease', 'stroke', 'all'],
        default='all',
        help='Disease to train (for train mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        run_api(args)
    elif args.mode == 'train':
        run_training(args)
    elif args.mode == 'test':
        run_tests()


def run_api(args):
    """Run the FastAPI application."""
    try:
        import uvicorn
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         Healthcare Prediction System - API Server             ║
╠═══════════════════════════════════════════════════════════════╣
║  Host: {args.host:<54} ║
║  Port: {args.port:<54} ║
║  Workers: {args.workers:<51} ║
║  Reload: {str(args.reload):<52} ║
╚═══════════════════════════════════════════════════════════════╝

Starting server...
        """)
        
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload
        )
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


def run_training(args):
    """Run model training."""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║        Healthcare Prediction System - Model Training          ║
╠═══════════════════════════════════════════════════════════════╣
║  Disease(s): {args.disease:<49} ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        from ml.training import train_disease_model
        from core.logging_config import get_logger
        
        logger = get_logger(__name__)
        
        diseases = ['diabetes', 'heart_disease', 'stroke'] if args.disease == 'all' else [args.disease]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_paths = {
            'diabetes': os.path.join(base_dir, 'data', 'raw', 'diabetes_data.csv'),
            'heart_disease': os.path.join(base_dir, 'data', 'raw', 'heart_disease_data.csv'),
            'stroke': os.path.join(base_dir, 'data', 'raw', 'stroke_data.csv'),
        }
        output_dir = os.path.join(base_dir, 'models')
        
        for disease in diseases:
            print(f"\n{'='*60}")
            print(f"Training {disease.upper()} model...")
            print('='*60)
            
            try:
                data_path = data_paths[disease]
                result = train_disease_model(disease, data_path, output_dir)
                
                print(f"\n✅ Training completed for {disease}")
                print(f"   Model: {result.model_name}")
                print(f"   CV Recall: {result.cv_mean.get('recall', 0.0):.3f}")
                print(f"   CV Precision: {result.cv_mean.get('precision', 0.0):.3f}")
                print(f"   Threshold: {result.optimal_threshold:.3f}")
                
            except Exception as e:
                print(f"\n❌ Error training {disease}: {e}")
                logger.error(f"Training failed for {disease}", exc_info=True)
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print('='*60)
        
    except ImportError as e:
        print(f"Error: Missing dependencies. {e}")
        print("Run: pip install -r requirements-production.txt")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          Healthcare Prediction System - Test Suite            ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        import pytest
        
        # Run tests with verbose output
        exit_code = pytest.main([
            'tests/',
            '-v',
            '--tb=short',
            '-x'  # Stop on first failure
        ])
        
        sys.exit(exit_code)
        
    except ImportError:
        print("Error: pytest not installed. Run: pip install pytest")
        sys.exit(1)


if __name__ == '__main__':
    main()
