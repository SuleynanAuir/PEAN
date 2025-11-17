"""
Academic-level visualization utilities for PEAN training
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os


def plot_training_curves(metrics_history, save_path='./ckpt/training_curves.png'):
    """
    Generate comprehensive training curves with academic style
    
    Args:
        metrics_history: Dictionary containing training metrics
        save_path: Path to save the figure
    """
    # Set academic style with fallback
    try:
        plt.style.use('seaborn-paper')
    except:
        try:
            plt.style.use('seaborn-v0_8-paper')
        except:
            plt.style.use('default')
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    iterations = metrics_history.get('iterations', [])
    if len(iterations) == 0:
        return
    
    # 1. Loss curve (top-left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    loss = metrics_history.get('loss', [])
    if len(loss) > 0:
        ax1.plot(iterations, loss, 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
        # Add moving average
        if len(loss) > 50:
            window = min(50, len(loss) // 10)
            loss_smooth = np.convolve(loss, np.ones(window)/window, mode='valid')
            ax1.plot(iterations[window-1:], loss_smooth, 'r-', linewidth=2, label=f'MA({window})')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Learning Rate (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    lr = metrics_history.get('learning_rate', [])
    if len(lr) > 0:
        ax2.plot(iterations, lr, 'g-', linewidth=1.5)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule', fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. PSNR curves for all datasets (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    datasets = ['easy', 'medium', 'hard']
    
    for idx, dataset in enumerate(datasets):
        psnr_key = f'psnr_{dataset}'
        if psnr_key in metrics_history and len(metrics_history[psnr_key]) > 0:
            # Get corresponding iterations for validation
            val_iters = iterations[::len(iterations)//len(metrics_history[psnr_key])][:len(metrics_history[psnr_key])]
            ax3.plot(val_iters, metrics_history[psnr_key], 
                    color=colors[idx], marker=markers[idx], linewidth=2, 
                    markersize=4, label=dataset.capitalize(), alpha=0.8)
    
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('PSNR (dB)')
    ax3.set_title('PSNR on Test Sets', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. SSIM curves for all datasets (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, dataset in enumerate(datasets):
        ssim_key = f'ssim_{dataset}'
        if ssim_key in metrics_history and len(metrics_history[ssim_key]) > 0:
            val_iters = iterations[::len(iterations)//len(metrics_history[ssim_key])][:len(metrics_history[ssim_key])]
            ax4.plot(val_iters, metrics_history[ssim_key], 
                    color=colors[idx], marker=markers[idx], linewidth=2,
                    markersize=4, label=dataset.capitalize(), alpha=0.8)
    
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('SSIM')
    ax4.set_title('SSIM on Test Sets', fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. Accuracy curves (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    for idx, dataset in enumerate(datasets):
        acc_key = f'acc_aster_{dataset}'
        if acc_key in metrics_history and len(metrics_history[acc_key]) > 0:
            val_iters = iterations[::len(iterations)//len(metrics_history[acc_key])][:len(metrics_history[acc_key])]
            ax5.plot(val_iters, [a*100 for a in metrics_history[acc_key]], 
                    color=colors[idx], marker=markers[idx], linewidth=2,
                    markersize=4, label=dataset.capitalize(), alpha=0.8)
    
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Recognition Accuracy (ASTER)', fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 6. Performance comparison bar chart (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    latest_metrics = []
    for dataset in datasets:
        psnr = metrics_history.get(f'psnr_{dataset}', [])
        latest_metrics.append(psnr[-1] if psnr else 0)
    
    if any(latest_metrics):
        bars = ax6.bar(datasets, latest_metrics, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('PSNR (dB)')
        ax6.set_title('Latest PSNR Comparison', fontweight='bold')
        ax6.set_ylim([0, max(latest_metrics) * 1.2])
        for bar, val in zip(bars, latest_metrics):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 7. SSIM comparison bar chart (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    latest_ssim = []
    for dataset in datasets:
        ssim = metrics_history.get(f'ssim_{dataset}', [])
        latest_ssim.append(ssim[-1] if ssim else 0)
    
    if any(latest_ssim):
        bars = ax7.bar(datasets, latest_ssim, color=colors, alpha=0.7, edgecolor='black')
        ax7.set_ylabel('SSIM')
        ax7.set_title('Latest SSIM Comparison', fontweight='bold')
        ax7.set_ylim([0, 1.0])
        for bar, val in zip(bars, latest_ssim):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax7.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 8. Accuracy comparison bar chart (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    latest_acc = []
    for dataset in datasets:
        acc = metrics_history.get(f'acc_aster_{dataset}', [])
        latest_acc.append(acc[-1]*100 if acc else 0)
    
    if any(latest_acc):
        bars = ax8.bar(datasets, latest_acc, color=colors, alpha=0.7, edgecolor='black')
        ax8.set_ylabel('Accuracy (%)')
        ax8.set_title('Latest Accuracy Comparison', fontweight='bold')
        ax8.set_ylim([0, 100])
        for bar, val in zip(bars, latest_acc):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax8.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add overall title
    fig.suptitle('PEAN Training Progress Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Training curves saved to: {save_path}')


def plot_comparison_table(metrics_history, save_path='./ckpt/metrics_table.png'):
    """
    Create a professional metrics comparison table
    """
    datasets = ['easy', 'medium', 'hard']
    
    # Gather latest metrics
    data = []
    for dataset in datasets:
        psnr = metrics_history.get(f'psnr_{dataset}', [])
        ssim = metrics_history.get(f'ssim_{dataset}', [])
        acc = metrics_history.get(f'acc_aster_{dataset}', [])
        
        data.append([
            dataset.capitalize(),
            f'{psnr[-1]:.2f}' if psnr else 'N/A',
            f'{ssim[-1]:.4f}' if ssim else 'N/A',
            f'{acc[-1]*100:.2f}%' if acc else 'N/A'
        ])
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data,
                     colLabels=['Dataset', 'PSNR (dB)', 'SSIM', 'Accuracy'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(data)+1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('Latest Validation Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Metrics table saved to: {save_path}')


def plot_diffusion_features(diffusion_states, timesteps, save_path='./ckpt/visualizations/diffusion_process.png'):
    """
    Visualize the diffusion model denoising process
    
    Args:
        diffusion_states: List of tensors representing diffusion states at different timesteps
                         Each tensor shape: (C, L) where C=channels, L=sequence length
        timesteps: List of timestep values (e.g., [1000, 750, 500, 250, 100, 0])
        save_path: Path to save the visualization
    """
    try:
        plt.style.use('seaborn-paper')
    except:
        try:
            plt.style.use('seaborn-v0_8-paper')
        except:
            plt.style.use('default')
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.titlesize': 14
    })
    
    num_steps = len(diffusion_states)
    fig, axes = plt.subplots(2, num_steps, figsize=(3*num_steps, 6))
    
    if num_steps == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (state, t) in enumerate(zip(diffusion_states, timesteps)):
        # Convert tensor to numpy
        if hasattr(state, 'cpu'):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = np.array(state)
        
        # Take first sample if batched
        if len(state_np.shape) == 3:
            state_np = state_np[0]
        
        # Row 1: Feature heatmap
        im1 = axes[0, idx].imshow(state_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0, idx].set_title(f't = {t}', fontweight='bold')
        axes[0, idx].set_ylabel('Feature Channel')
        axes[0, idx].set_xlabel('Sequence Position')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)
        
        # Row 2: Mean activation per channel
        mean_activation = state_np.mean(axis=1)
        axes[1, idx].bar(range(len(mean_activation)), mean_activation, color='steelblue', alpha=0.7)
        axes[1, idx].set_xlabel('Channel Index')
        axes[1, idx].set_ylabel('Mean Activation')
        axes[1, idx].set_title('Channel Activations', fontweight='bold')
        axes[1, idx].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    fig.suptitle('TPEM Diffusion Model: Text Prior Denoising Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Diffusion features saved to: {save_path}')


def plot_prediction_examples(examples, save_path='./ckpt/visualizations/prediction_examples.png'):
    """
    Visualize correct and incorrect prediction examples
    
    Args:
        examples: Dictionary with keys 'correct' and 'incorrect', each containing list of:
                 {'lr': tensor, 'sr': tensor, 'hr': tensor, 'pred': str, 'label': str}
        save_path: Path to save the visualization
    """
    try:
        plt.style.use('seaborn-paper')
    except:
        try:
            plt.style.use('seaborn-v0_8-paper')
        except:
            plt.style.use('default')
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'figure.titlesize': 14
    })
    
    correct_examples = examples.get('correct', [])[:5]  # Top 5 correct
    incorrect_examples = examples.get('incorrect', [])[:5]  # Top 5 incorrect
    
    num_correct = len(correct_examples)
    num_incorrect = len(incorrect_examples)
    
    if num_correct + num_incorrect == 0:
        print("No examples to visualize")
        return
    
    # Create figure with two sections
    fig = plt.figure(figsize=(18, 4*(num_correct + num_incorrect)))
    gs = fig.add_gridspec(num_correct + num_incorrect, 3, hspace=0.4, wspace=0.2)
    
    def add_example(row_idx, example, is_correct):
        """Helper function to add one example row"""
        # LR image
        ax_lr = fig.add_subplot(gs[row_idx, 0])
        lr_img = tensor_to_image(example['lr'])
        ax_lr.imshow(lr_img, cmap='gray' if len(lr_img.shape) == 2 else None)
        ax_lr.set_title('LR Input', fontweight='bold')
        ax_lr.axis('off')
        
        # SR image
        ax_sr = fig.add_subplot(gs[row_idx, 1])
        sr_img = tensor_to_image(example['sr'])
        ax_sr.imshow(sr_img, cmap='gray' if len(sr_img.shape) == 2 else None)
        color = 'green' if is_correct else 'red'
        ax_sr.set_title(f'SR Output (Pred: {example["pred"]})', 
                       fontweight='bold', color=color)
        ax_sr.axis('off')
        
        # HR ground truth
        ax_hr = fig.add_subplot(gs[row_idx, 2])
        hr_img = tensor_to_image(example['hr'])
        ax_hr.imshow(hr_img, cmap='gray' if len(hr_img.shape) == 2 else None)
        ax_hr.set_title(f'HR Ground Truth (Label: {example["label"]})', fontweight='bold')
        ax_hr.axis('off')
    
    # Add correct examples
    for idx, example in enumerate(correct_examples):
        add_example(idx, example, is_correct=True)
    
    # Add incorrect examples
    for idx, example in enumerate(incorrect_examples):
        add_example(num_correct + idx, example, is_correct=False)
    
    # Add section labels
    if num_correct > 0:
        fig.text(0.02, 1 - (num_correct/2)/(num_correct + num_incorrect), 
                'Correct Predictions ✓', fontsize=14, fontweight='bold', 
                color='green', va='center', rotation=90)
    
    if num_incorrect > 0:
        fig.text(0.02, (num_correct/2 + num_incorrect/2)/(num_correct + num_incorrect),
                'Incorrect Predictions ✗', fontsize=14, fontweight='bold',
                color='red', va='center', rotation=90)
    
    fig.suptitle('Prediction Analysis: Correct vs. Incorrect Cases', 
                fontsize=16, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Prediction examples saved to: {save_path}')


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor to numpy image for visualization
    
    Args:
        tensor: PyTorch tensor (C, H, W) or (H, W)
    
    Returns:
        numpy array suitable for matplotlib imshow
    """
    if hasattr(tensor, 'cpu'):
        img = tensor.detach().cpu().numpy()
    else:
        img = np.array(tensor)
    
    # Handle different tensor formats
    if len(img.shape) == 3:  # (C, H, W)
        if img.shape[0] == 1:  # Grayscale
            img = img[0]
        elif img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
    
    # Normalize to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0
    
    # Clip values
    img = np.clip(img, 0, 1)
    
    return img
