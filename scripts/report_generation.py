"""
Generate performance reports and visualizations for the Q&A chatbot system.
"""
import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from scripts.config import PROJECT_ROOT, EVALUATION_METRICS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_evaluation_report(
    eval_results: Dict[str, Any],
    report_name: Optional[str] = None,
    save_format: str = "all"
) -> Dict[str, Any]:
    """
    Generate evaluation report from RAGAS evaluation results.
    
    Args:
        eval_results: Dictionary containing evaluation results
        report_name: Optional name for the report (default: uses timestamp)
        save_format: Format to save report ("json", "html", "pdf", or "all")
        
    Returns:
        Dictionary with paths to saved reports
    """
    if not report_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"evaluation_report_{timestamp}"
    
    # Ensure the report_name doesn't have file extensions
    report_name = os.path.splitext(report_name)[0]
    
    logger.info(f"Generating evaluation report: {report_name}")
    
    try:
        # Convert results to DataFrame for easier processing
        metrics_data = {}
        for agent_name, results in eval_results.items():
            agent_metrics = {}
            for metric_name in EVALUATION_METRICS:
                if metric_name in results:
                    agent_metrics[metric_name] = results[metric_name]
            metrics_data[agent_name] = agent_metrics
        
        df = pd.DataFrame(metrics_data).T
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        
        # Radar chart for metric comparison
        if len(df.columns) > 2:
            ax = plt.subplot(2, 2, 1, polar=True)
            categories = df.columns.tolist()
            N = len(categories)
            
            # Create angle values for each metric
            angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot each agent
            for agent_name, metrics in df.iterrows():
                values = metrics.values.tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=agent_name)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels and title
            plt.xticks(angles[:-1], categories)
            plt.title("Agent Performance Comparison")
            plt.legend(loc='upper right')
        
        # Bar chart for metric comparison
        ax = plt.subplot(2, 2, 2)
        df.plot(kind='bar', ax=ax)
        ax.set_title("Performance Metrics by Agent")
        ax.set_ylabel("Score")
        ax.set_xlabel("Agent")
        
        # Heatmap for all metrics
        ax = plt.subplot(2, 2, 3)
        sns.heatmap(df, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("Performance Heatmap")
        
        # Detailed metrics table
        ax = plt.subplot(2, 2, 4)
        ax.axis('off')
        table_data = []
        for agent, metrics in metrics_data.items():
            for metric, value in metrics.items():
                table_data.append([agent, metric, f"{value:.4f}"])
        
        table = ax.table(
            cellText=table_data,
            colLabels=["Agent", "Metric", "Value"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title("Detailed Metrics Table")
        
        plt.tight_layout()
        
        # Save reports in different formats
        saved_paths = {}
        
        # Save JSON data
        if save_format in ["json", "all"]:
            json_path = REPORTS_DIR / f"{report_name}.json"
            with open(json_path, 'w') as f:
                json.dump(eval_results, f, indent=4)
            saved_paths["json"] = str(json_path)
        
        # Save figure
        fig_path = REPORTS_DIR / f"{report_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        saved_paths["figure"] = str(fig_path)
        
        # Save HTML report
        if save_format in ["html", "all"]:
            html_path = REPORTS_DIR / f"{report_name}.html"
            
            # Create simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Evaluation Report - {report_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metrics-table {{ margin-bottom: 40px; }}
                    .visualization {{ text-align: center; margin: 20px 0; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Performance Metrics</h2>
                <div class="metrics-table">
                    <table>
                        <tr>
                            <th>Agent</th>
                            {''.join([f'<th>{metric}</th>' for metric in EVALUATION_METRICS])}
                        </tr>
                        {''.join([
                            f'<tr><td>{agent}</td>' + 
                            ''.join([f'<td>{results.get(metric, "N/A"):.4f}</td>' if isinstance(results.get(metric), (int, float)) else f'<td>{results.get(metric, "N/A")}</td>' for metric in EVALUATION_METRICS]) + 
                            '</tr>' 
                            for agent, results in eval_results.items()
                        ])}
                    </table>
                </div>
                
                <h2>Visualizations</h2>
                <div class="visualization">
                    <img src="{os.path.basename(fig_path)}" alt="Performance Visualization">
                </div>
                
                <h2>Raw Data</h2>
                <pre>{json.dumps(eval_results, indent=4)}</pre>
            </body>
            </html>
            """
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            saved_paths["html"] = str(html_path)
        
        # Save PDF report (requires additional libraries)
        if save_format in ["pdf", "all"]:
            try:
                from weasyprint import HTML
                pdf_path = REPORTS_DIR / f"{report_name}.pdf"
                HTML(string=html_content).write_pdf(pdf_path)
                saved_paths["pdf"] = str(pdf_path)
            except ImportError:
                logger.warning("weasyprint not installed. PDF report not generated.")
                saved_paths["pdf"] = "Not generated (weasyprint not installed)"
        
        plt.close()
        logger.info(f"Report generated successfully: {saved_paths}")
        return saved_paths
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return {"error": str(e)}

def generate_usage_report(
    usage_data: List[Dict[str, Any]],
    report_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate usage report for the chatbot system.
    
    Args:
        usage_data: List of usage data entries
        report_name: Optional name for the report (default: uses timestamp)
        
    Returns:
        Dictionary with paths to saved reports
    """
    if not report_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"usage_report_{timestamp}"
    
    logger.info(f"Generating usage report: {report_name}")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(usage_data)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Query distribution by agent type
        if 'agent_type' in df.columns:
            ax1 = plt.subplot(2, 2, 1)
            agent_counts = df['agent_type'].value_counts()
            agent_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
            ax1.set_title('Query Distribution by Agent Type')
            ax1.set_ylabel('')
        
        # Response time distribution
        if 'response_time' in df.columns:
            ax2 = plt.subplot(2, 2, 2)
            sns.histplot(df['response_time'], kde=True, ax=ax2)
            ax2.set_title('Response Time Distribution')
            ax2.set_xlabel('Response Time (seconds)')
        
        # Success rate over time
        if 'timestamp' in df.columns and 'success' in df.columns:
            ax3 = plt.subplot(2, 2, 3)
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by date and calculate success rate
            df['date'] = df['timestamp'].dt.date
            success_rate = df.groupby('date')['success'].mean()
            success_rate.plot(marker='o', ax=ax3)
            ax3.set_title('Success Rate Over Time')
            ax3.set_ylabel('Success Rate')
            ax3.set_ylim([0, 1])
        
        # Top query types or keywords
        if 'query' in df.columns:
            ax4 = plt.subplot(2, 2, 4)
            # Simple keyword extraction (could be more sophisticated)
            df['query_length'] = df['query'].str.len()
            query_length_bins = pd.cut(df['query_length'], bins=[0, 50, 100, 150, 200, 1000])
            query_length_counts = query_length_bins.value_counts().sort_index()
            query_length_counts.plot(kind='bar', ax=ax4)
            ax4.set_title('Query Length Distribution')
            ax4.set_xlabel('Query Length (characters)')
        
        plt.tight_layout()
        
        # Save reports
        fig_path = REPORTS_DIR / f"{report_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Save CSV data
        csv_path = REPORTS_DIR / f"{report_name}.csv"
        df.to_csv(csv_path, index=False)
        
        plt.close()
        
        return {
            "figure": str(fig_path),
            "data": str(csv_path)
        }
        
    except Exception as e:
        logger.error(f"Error generating usage report: {str(e)}")
        return {"error": str(e)}

def generate_system_health_report(
    metrics: Dict[str, Any],
    report_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate system health and performance report.
    
    Args:
        metrics: Dictionary containing system metrics
        report_name: Optional name for the report
        
    Returns:
        Dictionary with paths to saved reports
    """
    if not report_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"system_health_{timestamp}"
    
    logger.info(f"Generating system health report: {report_name}")
    
    try:
        # Create visualizations for system metrics
        plt.figure(figsize=(12, 8))
        
        # Database size over time
        if 'db_size_history' in metrics:
            ax1 = plt.subplot(2, 2, 1)
            db_sizes = pd.DataFrame(metrics['db_size_history'])
            db_sizes.plot(x='timestamp', y=['vector_db_size', 'sql_db_size'], ax=ax1)
            ax1.set_title('Database Size Over Time')
            ax1.set_ylabel('Size (MB)')
        
        # Response time trends
        if 'response_time_history' in metrics:
            ax2 = plt.subplot(2, 2, 2)
            resp_times = pd.DataFrame(metrics['response_time_history'])
            resp_times.plot(x='timestamp', y=['rag_agent', 'sql_agent', 'primary_agent'], ax=ax2)
            ax2.set_title('Response Time Trends')
            ax2.set_ylabel('Time (seconds)')
        
        # Memory usage
        if 'memory_usage' in metrics:
            ax3 = plt.subplot(2, 2, 3)
            memory = pd.DataFrame(metrics['memory_usage'])
            memory.plot(kind='area', ax=ax3)
            ax3.set_title('Memory Usage')
            ax3.set_ylabel('Memory (MB)')
        
        # Error rates
        if 'error_rates' in metrics:
            ax4 = plt.subplot(2, 2, 4)
            error_rates = pd.DataFrame(metrics['error_rates'])
            error_rates.plot(kind='bar', ax=ax4)
            ax4.set_title('Error Rates by Component')
            ax4.set_ylabel('Error Rate (%)')
        
        plt.tight_layout()
        
        # Save report
        fig_path = REPORTS_DIR / f"{report_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Save data
        json_path = REPORTS_DIR / f"{report_name}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        plt.close()
        
        return {
            "figure": str(fig_path),
            "data": str(json_path)
        }
        
    except Exception as e:
        logger.error(f"Error generating system health report: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test with sample data
    sample_eval_results = {
        "primary_agent": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.78,
            "context_precision": 0.92,
            "context_recall": 0.76
        },
        "rag_agent": {
            "faithfulness": 0.82,
            "answer_relevancy": 0.75,
            "context_precision": 0.88,
            "context_recall": 0.81
        },
        "sql_agent": {
            "faithfulness": 0.91,
            "answer_relevancy": 0.85,
            "context_precision": 0.94,
            "context_recall": 0.72
        }
    }
    
    # Generate sample evaluation report
    report_paths = generate_evaluation_report(
        sample_eval_results,
        report_name="sample_evaluation",
        save_format="all"
    )
    
    print(f"Report saved to: {report_paths}")