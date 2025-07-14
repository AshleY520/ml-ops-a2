import sys
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_experiment_runs():
    """Get all completed runs from experiments"""
    try:
        logger.info("Waiting for MLflow to index recent runs...")
        time.sleep(2)
        
        current_dir = os.getcwd()
        mlruns_path = os.path.join(current_dir, "mlruns")
        tracking_uri = f"file://{mlruns_path}"
        
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Current tracking URI: {mlflow.get_tracking_uri()}")
        
        experiments = mlflow.search_experiments()
        logger.info(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            logger.info(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        logger.info("Searching across all experiments...")
        all_runs = mlflow.search_runs(filter_string="status = 'FINISHED'")
        logger.info(f"Found {len(all_runs)} finished runs across all experiments")
        
        if all_runs.empty:
            logger.warning("No runs found!")
            return pd.DataFrame()
        
        logger.info("Available columns:")
        param_columns = [col for col in all_runs.columns if col.startswith('params.')]
        metric_columns = [col for col in all_runs.columns if col.startswith('metrics.')]
        
        logger.info(f"Parameter columns: {param_columns}")
        logger.info(f"Metric columns: {metric_columns}")
        
        feature_version_cols = [col for col in param_columns if 'feature_version' in col]
        n_estimators_cols = [col for col in param_columns if 'n_estimators' in col]
        
        if feature_version_cols or n_estimators_cols:
            logger.info(f"Found feature version columns: {feature_version_cols}")
            logger.info(f"Found n_estimators columns: {n_estimators_cols}")
            
            training_runs = all_runs.copy()
            
            has_training_params = pd.Series(False, index=all_runs.index)
            
            for col in feature_version_cols + n_estimators_cols:
                has_training_params |= all_runs[col].notna()
            
            training_runs = all_runs[has_training_params]
            logger.info(f"Training runs found: {len(training_runs)}")
            
        else:
            logger.warning("No training parameters found - using all runs")
            training_runs = all_runs
        
        if not training_runs.empty:
            logger.info(f"Final result - Total runs: {len(training_runs)}")
            
            if 'status' in training_runs.columns:
                logger.info(f"Run statuses: {training_runs['status'].value_counts().to_dict()}")
            
            logger.info("\nDetailed run information:")
            for idx, row in training_runs.iterrows():
                logger.info(f"Run {idx + 1}:")
                logger.info(f"  ID: {row.get('run_id', 'N/A')[:8]}...")
                logger.info(f"  Status: {row.get('status', 'N/A')}")
                
                feature_version = "N/A"
                n_estimators = "N/A"
                test_r2 = "N/A"
                test_mse = "N/A"
                emissions = "N/A"
                
                for col in feature_version_cols:
                    if pd.notna(row.get(col)):
                        feature_version = row[col]
                        break
                
                for col in n_estimators_cols:
                    if pd.notna(row.get(col)):
                        n_estimators = row[col]
                        break
                
                for col in metric_columns:
                    if 'test_r2' in col and pd.notna(row.get(col)):
                        test_r2 = row[col]
                    elif 'test_mse' in col and pd.notna(row.get(col)):
                        test_mse = row[col]
                    elif 'carbon_emissions_kg' in col and pd.notna(row.get(col)):
                        emissions = row[col]
                
                logger.info(f"  Feature Version: {feature_version}")
                logger.info(f"  N Estimators: {n_estimators}")
                logger.info(f"  Test R²: {test_r2}")
                logger.info(f"  Test MSE: {test_mse}")
                logger.info(f"  Emissions: {emissions}")
        else:
            logger.warning("No training runs found!")
        
        return training_runs
        
    except Exception as e:
        logger.error(f"Error getting runs: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def extract_model_results(runs_df):
    """Extract model results from MLflow runs with new parameter naming"""
    results = []
    
    for idx, run in runs_df.iterrows():
        run_data = {
            'run_id': run['run_id'],
            'status': run['status'],
            'start_time': run.get('start_time', ''),
            'end_time': run.get('end_time', '')
        }
        
        param_columns = [col for col in run.index if col.startswith('params.')]
        metric_columns = [col for col in run.index if col.startswith('metrics.')]
        
        model_configs = {}
        
        for param_col in param_columns:
            param_value = run[param_col]
            if pd.notna(param_value):
                if 'feature_version' in param_col:
                    suffix = param_col.replace('params.feature_version', '')
                    if suffix and suffix != '':
                        config_key = suffix
                        if config_key not in model_configs:
                            model_configs[config_key] = {}
                        model_configs[config_key]['feature_version'] = param_value
                    else:
                        model_configs['_default'] = model_configs.get('_default', {})
                        model_configs['_default']['feature_version'] = param_value
                        
                elif 'n_estimators' in param_col:
                    suffix = param_col.replace('params.n_estimators', '')
                    if suffix and suffix != '':
                        config_key = suffix
                        if config_key not in model_configs:
                            model_configs[config_key] = {}
                        model_configs[config_key]['n_estimators'] = param_value
                    else:
                        model_configs['_default'] = model_configs.get('_default', {})
                        model_configs['_default']['n_estimators'] = param_value
                        
                elif 'max_depth' in param_col:
                    suffix = param_col.replace('params.max_depth', '')
                    if suffix and suffix != '':
                        config_key = suffix
                        if config_key not in model_configs:
                            model_configs[config_key] = {}
                        model_configs[config_key]['max_depth'] = param_value
                    else:
                        model_configs['_default'] = model_configs.get('_default', {})
                        model_configs['_default']['max_depth'] = param_value
   
        for config_key, config in model_configs.items():
            if config_key == '_default':
                test_r2 = run.get('metrics.test_r2', np.nan)
                test_mse = run.get('metrics.test_mse', np.nan)
                carbon = run.get('metrics.carbon_emissions_kg', np.nan)
                features_count = run.get('metrics.features_count', np.nan)
            else:
                test_r2 = run.get(f'metrics.test_r2{config_key}', np.nan)
                test_mse = run.get(f'metrics.test_mse{config_key}', np.nan)
                carbon = run.get(f'metrics.carbon_emissions_kg{config_key}', np.nan)
                features_count = run.get(f'metrics.features_count{config_key}', np.nan)
            
            if 'feature_version' in config and 'n_estimators' in config:
                result = {
                    'run_id': run['run_id'],
                    'config_suffix': config_key,
                    'feature_version': config.get('feature_version', 'Unknown'),
                    'n_estimators': config.get('n_estimators', 'Unknown'),
                    'max_depth': config.get('max_depth', 'Unknown'),
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'carbon_emissions_kg': carbon,
                    'features_count': features_count,
                    'status': run['status']
                }
                results.append(result)
    
    return pd.DataFrame(results)

def quantitative_analysis(runs_df):
    """Quantitative comparison of model metrics"""
    if runs_df.empty:
        logger.info("No runs found for analysis")
        return
    
    logger.info("\n" + "="*50)
    logger.info("QUANTITATIVE ANALYSIS")
    logger.info("="*50)
    
    results_df = extract_model_results(runs_df)
    
    if results_df.empty:
        logger.info("No model configurations found for analysis")
        return
    
    # Summary by feature version
    logger.info("\nPerformance by Feature Version:")
    
    for version in sorted(results_df['feature_version'].unique()):
        version_data = results_df[results_df['feature_version'] == version]
        valid_r2_data = version_data.dropna(subset=['test_r2'])
        
        if not valid_r2_data.empty:
            avg_r2 = valid_r2_data['test_r2'].mean()
            std_r2 = valid_r2_data['test_r2'].std()
            count = len(valid_r2_data)
            
            logger.info(f"\nVersion {version}:")
            logger.info(f"  Count: {count} models")
            logger.info(f"  R² - Mean: {avg_r2:.4f}, Std: {std_r2:.4f}")
            
            # MSE statistics
            valid_mse_data = version_data.dropna(subset=['test_mse'])
            if not valid_mse_data.empty:
                avg_mse = valid_mse_data['test_mse'].mean()
                std_mse = valid_mse_data['test_mse'].std()
                logger.info(f"  MSE - Mean: {avg_mse:.4f}, Std: {std_mse:.4f}")
            
            # Carbon emissions
            valid_carbon_data = version_data.dropna(subset=['carbon_emissions_kg'])
            if not valid_carbon_data.empty:
                total_carbon = valid_carbon_data['carbon_emissions_kg'].sum()
                avg_carbon = valid_carbon_data['carbon_emissions_kg'].mean()
                logger.info(f"  Carbon - Total: {total_carbon:.6f} kg, Avg: {avg_carbon:.6f} kg")
    
    # Best model
    valid_performance_data = results_df.dropna(subset=['test_r2'])
    if not valid_performance_data.empty:
        best_model = valid_performance_data.loc[valid_performance_data['test_r2'].idxmax()]
        logger.info(f"\n🏆 Best Model:")
        logger.info(f"  Feature Version: {best_model['feature_version']}")
        logger.info(f"  N Estimators: {best_model['n_estimators']}")
        logger.info(f"  Max Depth: {best_model['max_depth']}")
        logger.info(f"  Test R²: {best_model['test_r2']:.4f}")
        logger.info(f"  Test MSE: {best_model['test_mse']:.4f}")
        logger.info(f"  Features: {best_model['features_count']}")
        logger.info(f"  Carbon: {best_model['carbon_emissions_kg']:.6f} kg")

def qualitative_analysis(runs_df):
    """Generate comparison plots"""
    if runs_df.empty:
        logger.info("No data for plotting")
        return None
    
    logger.info("\nGenerating comparison plots...")
    
    results_df = extract_model_results(runs_df)
    
    if results_df.empty:
        logger.info("No model results for plotting")
        return None
    
    # Create plots directory
    Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Athletes ML Pipeline - Model Comparison', fontsize=16, fontweight='bold')
    
    try:
        # Plot 1: R² Score by Feature Version
        valid_r2_data = results_df.dropna(subset=['test_r2', 'feature_version'])
        if not valid_r2_data.empty:
            sns.boxplot(data=valid_r2_data, x='feature_version', y='test_r2', ax=axes[0,0])
            axes[0,0].set_title('R² Score by Feature Version')
            axes[0,0].set_xlabel('Feature Version')
            axes[0,0].set_ylabel('Test R² Score')
        else:
            axes[0,0].text(0.5, 0.5, 'No valid R² data available', ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('R² Score by Feature Version')
        
        # Plot 2: MSE by Feature Version
        valid_mse_data = results_df.dropna(subset=['test_mse', 'feature_version'])
        if not valid_mse_data.empty:
            sns.boxplot(data=valid_mse_data, x='feature_version', y='test_mse', ax=axes[0,1])
            axes[0,1].set_title('MSE by Feature Version')
            axes[0,1].set_xlabel('Feature Version')
            axes[0,1].set_ylabel('Test MSE')
        else:
            axes[0,1].text(0.5, 0.5, 'No valid MSE data available', ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('MSE by Feature Version')
        
        # Plot 3: Carbon Emissions
        valid_carbon_data = results_df.dropna(subset=['carbon_emissions_kg', 'feature_version'])
        if not valid_carbon_data.empty:
            sns.barplot(data=valid_carbon_data, x='feature_version', y='carbon_emissions_kg', ax=axes[1,0])
            axes[1,0].set_title('Carbon Emissions by Feature Version')
            axes[1,0].set_xlabel('Feature Version')
            axes[1,0].set_ylabel('Carbon Emissions (kg CO₂)')
        else:
            axes[1,0].text(0.5, 0.5, 'No valid emissions data available', ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Carbon Emissions by Feature Version')
        
        # Plot 4: Performance vs Emissions Scatter
        valid_scatter_data = results_df.dropna(subset=['test_r2', 'carbon_emissions_kg', 'feature_version'])
        if not valid_scatter_data.empty:
            for version in valid_scatter_data['feature_version'].unique():
                version_data = valid_scatter_data[valid_scatter_data['feature_version'] == version]
                axes[1,1].scatter(version_data['carbon_emissions_kg'], 
                                version_data['test_r2'], 
                                label=f'Version {version}', s=100, alpha=0.7)
            axes[1,1].set_xlabel('Carbon Emissions (kg CO₂)')
            axes[1,1].set_ylabel('Test R² Score')
            axes[1,1].set_title('Performance vs Carbon Footprint')
            axes[1,1].legend()
        else:
            axes[1,1].text(0.5, 0.5, 'No valid performance vs emissions data', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Performance vs Carbon Footprint')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "artifacts/plots/model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
        plt.close()
        return None

def carbon_emissions_analysis(runs_df):
    """Carbon emissions comparison"""
    if runs_df.empty:
        logger.info("No carbon emissions data found")
        return
    
    logger.info("\n" + "="*50)
    logger.info("CARBON EMISSIONS ANALYSIS")
    logger.info("="*50)
    
    results_df = extract_model_results(runs_df)
    
    if results_df.empty:
        logger.info("No model results for carbon analysis")
        return
    
    valid_emissions = results_df.dropna(subset=['carbon_emissions_kg'])
    
    if valid_emissions.empty:
        logger.info("No valid carbon emissions data found")
        return
    
    # Total emissions
    total_emissions = valid_emissions['carbon_emissions_kg'].sum()
    logger.info(f"\nTotal Carbon Emissions: {total_emissions:.6f} kg CO₂")
    
    # Emissions by feature version
    emissions_by_version = valid_emissions.groupby('feature_version')['carbon_emissions_kg'].agg(['sum', 'mean', 'count'])
    logger.info(f"\nEmissions by Feature Version:")
    for version in emissions_by_version.index:
        data = emissions_by_version.loc[version]
        logger.info(f"  Version {version}:")
        logger.info(f"    Total: {data['sum']:.6f} kg")
        logger.info(f"    Average: {data['mean']:.6f} kg")
        logger.info(f"    Count: {data['count']} models")
    
    # Most efficient model (best R²/emissions ratio)
    efficiency_data = results_df.dropna(subset=['test_r2', 'carbon_emissions_kg'])
    if not efficiency_data.empty:
        efficiency_data = efficiency_data.copy()
        efficiency_data['efficiency'] = efficiency_data['test_r2'] / efficiency_data['carbon_emissions_kg']
        most_efficient = efficiency_data.loc[efficiency_data['efficiency'].idxmax()]
        
        logger.info(f"\n🌱 Most Carbon Efficient Model:")
        logger.info(f"  Feature Version: {most_efficient['feature_version']}")
        logger.info(f"  Configuration: {most_efficient['n_estimators']} trees, depth {most_efficient['max_depth']}")
        logger.info(f"  Efficiency Ratio: {most_efficient['efficiency']:.2f} R²/kg")
        logger.info(f"  R² Score: {most_efficient['test_r2']:.4f}")
        logger.info(f"  Emissions: {most_efficient['carbon_emissions_kg']:.6f} kg")

def create_summary_report(runs_df):
    """Create a comprehensive summary report"""
    results_df = extract_model_results(runs_df)
    
    report_lines = []
    report_lines.append("# Athletes ML Pipeline - Experiment Summary\n\n")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_lines.append(f"Total Experiments: {len(results_df)}\n\n")
    
    if not results_df.empty:
        report_lines.append("## Feature Version Results\n\n")
        
        for version in sorted(results_df['feature_version'].unique()):
            version_data = results_df[results_df['feature_version'] == version]
            valid_r2_data = version_data.dropna(subset=['test_r2'])
            
            if not valid_r2_data.empty:
                avg_r2 = valid_r2_data['test_r2'].mean()
                max_r2 = valid_r2_data['test_r2'].max()
                
                report_lines.append(f"### Version {version}\n")
                report_lines.append(f"- Models: {len(valid_r2_data)}\n")
                report_lines.append(f"- Average R²: {avg_r2:.4f}\n")
                report_lines.append(f"- Best R²: {max_r2:.4f}\n")
                
                # Carbon info
                valid_carbon_data = version_data.dropna(subset=['carbon_emissions_kg'])
                if not valid_carbon_data.empty:
                    total_carbon = valid_carbon_data['carbon_emissions_kg'].sum()
                    report_lines.append(f"- Total Emissions: {total_carbon:.6f} kg CO₂\n")
                
                report_lines.append("\n")
        
        # Best result
        valid_performance_data = results_df.dropna(subset=['test_r2'])
        if not valid_performance_data.empty:
            best = valid_performance_data.loc[valid_performance_data['test_r2'].idxmax()]
            report_lines.append("## 🏆 Best Performance\n\n")
            report_lines.append(f"- Feature Version: {best['feature_version']}\n")
            report_lines.append(f"- R² Score: {best['test_r2']:.4f}\n")
            report_lines.append(f"- Configuration: {best['n_estimators']} estimators, depth {best['max_depth']}\n")
            report_lines.append(f"- Features: {best['features_count']}\n")
            report_lines.append(f"- Carbon Footprint: {best['carbon_emissions_kg']:.6f} kg CO₂\n\n")
        
        valid_carbon_data = results_df.dropna(subset=['carbon_emissions_kg'])
        if not valid_carbon_data.empty:
            total_emissions = valid_carbon_data['carbon_emissions_kg'].sum()
            report_lines.append("## 🌱 Environmental Impact\n\n")
            report_lines.append(f"- Total Carbon Emissions: {total_emissions:.6f} kg CO₂\n")
            report_lines.append(f"- Average per Model: {valid_carbon_data['carbon_emissions_kg'].mean():.6f} kg CO₂\n")
    else:
        report_lines.append("## No Results\n\n")
        report_lines.append("- No model results found for analysis\n")
    
    # Save report
    report_text = "".join(report_lines)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    report_path = "artifacts/experiment_summary.md"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    if not results_df.empty:
        csv_path = "artifacts/experiment_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results CSV saved to {csv_path}")
    
    logger.info(f"Summary report saved to {report_path}")
    return report_path

def main():
    """Main evaluation pipeline"""
    logger.info("Starting model evaluation...")
    
    # Get experiment runs
    runs_df = get_experiment_runs()
    
    if runs_df.empty:
        logger.warning("No completed runs found. Please run experiments first.")
        return
    
    logger.info(f"Found {len(runs_df)} runs for analysis")
    
    results_df = extract_model_results(runs_df)
    logger.info(f"Extracted {len(results_df)} model configurations")
    
    if results_df.empty:
        logger.warning("No model configurations found for analysis")
        return
    
    # Quantitative analysis
    quantitative_analysis(runs_df)
    
    # Qualitative analysis (plots)
    plot_path = qualitative_analysis(runs_df)
    
    # Carbon emissions analysis
    carbon_emissions_analysis(runs_df)
    
    # Create summary report
    report_path = create_summary_report(runs_df)
    
    # Log summary metrics only (without artifacts to avoid URI issues)
    try:
        with mlflow.start_run(run_name="evaluation_summary"):
            # Log summary metrics
            metrics = {
                "total_experiments": len(results_df)
            }
            
            valid_performance_data = results_df.dropna(subset=['test_r2'])
            if not valid_performance_data.empty:
                metrics["best_r2"] = valid_performance_data['test_r2'].max()
                metrics["avg_r2"] = valid_performance_data['test_r2'].mean()
            
            valid_carbon_data = results_df.dropna(subset=['carbon_emissions_kg'])
            if not valid_carbon_data.empty:
                metrics["total_emissions"] = valid_carbon_data['carbon_emissions_kg'].sum()
                metrics["avg_emissions"] = valid_carbon_data['carbon_emissions_kg'].mean()
            
            mlflow.log_metrics(metrics)
            
            # Log summary information as parameters
            mlflow.log_params({
                "evaluation_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_runs_analyzed": len(runs_df),
                "total_models_analyzed": len(results_df)
            })
            
        logger.info("MLflow evaluation summary logged successfully")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        logger.info("Continuing without MLflow artifact logging...")

    logger.info("Evaluation completed!")
    if plot_path:
        logger.info(f"Plots: {plot_path}")
    logger.info(f"Report: {report_path}")
    
    # Display final summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Analysis completed for {len(results_df)} model configurations")
    if plot_path:
        logger.info(f"Plots saved to: {plot_path}")
    logger.info(f"Report saved to: {report_path}")
    
    # Performance summary
    valid_performance_data = results_df.dropna(subset=['test_r2'])
    if not valid_performance_data.empty:
        logger.info(f"Best R² Score: {valid_performance_data['test_r2'].max():.4f}")
        logger.info(f"Average R² Score: {valid_performance_data['test_r2'].mean():.4f}")
    else:
        logger.info("Best R² Score: No performance data available")
    
    # Carbon summary
    valid_carbon_data = results_df.dropna(subset=['carbon_emissions_kg'])
    if not valid_carbon_data.empty:
        logger.info(f"Total Carbon Emissions: {valid_carbon_data['carbon_emissions_kg'].sum():.6f} kg CO₂")
    else:
        logger.info("Total Carbon Emissions: No emissions data available")
    
    logger.info("="*50)

if __name__ == "__main__":
    main()