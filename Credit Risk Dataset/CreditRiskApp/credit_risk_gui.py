import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
from scipy import stats
import joblib
import skops.io as sio
import os
from datetime import datetime

class CreditRiskPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Risk Prediction System")
        self.root.geometry("750x850")
        
        # Load the saved model and Box-Cox lambdas
        self.load_model()
        
        # Define features and categories
        self.numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        self.categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade']
        
        self.home_ownership_options = ['MORTGAGE', 'OWN', 'RENT', 'OTHER']
        self.loan_intent_options = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
        self.loan_grade_options = ['A', 'B', 'C', 'D', 'E', 'Other']
        
        self.entries = {}
        self.combos = {}
        self.default_var = tk.BooleanVar()
        
        self.create_widgets()
        self.update_prediction_count()
    
    def load_model(self):
        """Load the trained model and preprocessing parameters"""
        try:
            # Load model
            if not os.path.exists("model.skops"):
                raise FileNotFoundError("model.skops not found")
            if not os.path.exists("lambdas.pkl"):
                raise FileNotFoundError("lambdas.pkl not found")
                
            unknown_types = sio.get_untrusted_types(file="model.skops")
            self.model = sio.load("model.skops", trusted=unknown_types)
            
            # Load Box-Cox transformation parameters
            self.lambdas = joblib.load('lambdas.pkl')
            self.boxcox_features = list(self.lambdas.keys())
            
        except FileNotFoundError as e:
            messagebox.showerror("File Error", 
                f"Required files not found: {e}\nPlease ensure 'model.skops' and 'lambdas.pkl' exist.")
            raise
        except Exception as e:
            messagebox.showerror("Loading Error", f"Error loading model files: {e}")
            raise
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame with scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Credit Risk Prediction System", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))
        
        # Statistics frame
        stats_frame = ttk.Frame(scrollable_frame)
        stats_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="Saved Predictions: 0", font=("Arial", 10))
        self.stats_label.pack(side=tk.LEFT, padx=(10, 20))
        
        # Instructions
        instructions = ttk.Label(scrollable_frame, 
                                text="Fill in the loan application details below:",
                                font=("Arial", 10))
        instructions.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Numeric features section
        ttk.Label(scrollable_frame, text="Numeric Information", 
                 font=("Arial", 12, "bold")).grid(row=3, column=0, columnspan=3, pady=(10, 10))
        
        row = 4
        for i, feature in enumerate(self.numeric_features):
            col = i % 3
            if col == 0:
                row += 1
                
            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            
            label_text = feature.replace('_', ' ').title()
            ttk.Label(frame, text=label_text, font=("Arial", 9)).pack(anchor="w")
            
            self.entries[feature] = ttk.Entry(frame, width=15)
            self.entries[feature].pack(anchor="w")
        
        # Categorical features section
        ttk.Label(scrollable_frame, text="Categorical Information", 
                 font=("Arial", 12, "bold")).grid(row=row+1, column=0, columnspan=3, pady=(20, 10))
        
        row += 2
        for i, feature in enumerate(self.categorical_features):
            col = i % 3
            if col == 0:
                row += 1
                
            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            
            label_text = feature.replace('_', ' ').title()
            ttk.Label(frame, text=label_text, font=("Arial", 9)).pack(anchor="w")
            
            if feature == 'person_home_ownership':
                options = self.home_ownership_options
            elif feature == 'loan_intent':
                options = self.loan_intent_options
            else:
                options = self.loan_grade_options
                
            self.combos[feature] = ttk.Combobox(frame, values=options, width=12, state="readonly")
            self.combos[feature].pack(anchor="w")
        
        # Binary feature
        row += 1
        binary_frame = ttk.Frame(scrollable_frame)
        binary_frame.grid(row=row, column=0, columnspan=3, pady=(15, 10))
        
        ttk.Label(binary_frame, text="Previous Credit Default:", 
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(binary_frame, variable=self.default_var).pack(side=tk.LEFT)
        ttk.Label(binary_frame, text="(Check if applicant had previous default)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Buttons frame
        row += 1
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        # Predict button
        predict_btn = ttk.Button(button_frame, text="Predict Risk", 
                                command=self.predict_loan_status, width=15)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = ttk.Button(button_frame, text="Clear All", 
                              command=self.clear_inputs, width=15)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Sample data button
        sample_btn = ttk.Button(button_frame, text="Load Sample Data", 
                               command=self.load_sample_data, width=15)
        sample_btn.pack(side=tk.LEFT, padx=5)
        
        # Export buttons
        export_csv_btn = ttk.Button(button_frame, text="Export CSV", 
                                   command=self.export_all_data_csv, width=12)
        export_csv_btn.pack(side=tk.LEFT, padx=5)
        
        export_pdf_btn = ttk.Button(button_frame, text="Export PDF", 
                                   command=self.export_all_data_pdf, width=12)
        export_pdf_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear predictions button
        clear_pred_btn = ttk.Button(button_frame, text="Clear All Predictions", 
                                   command=self.clear_all_predictions, width=15)
        clear_pred_btn.pack(side=tk.LEFT, padx=5)
        
        # Results section
        row += 1
        ttk.Label(scrollable_frame, text="Prediction Results", 
                 font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, pady=(20, 10))
        
        # Result frame
        row += 1
        result_frame = ttk.LabelFrame(scrollable_frame, text="Prediction Details", padding=10)
        result_frame.grid(row=row, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, width=75)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add sample data
        self.load_sample_data()
    
    def update_prediction_count(self):
        """Update the prediction count display"""
        try:
            if os.path.exists('loan_predictions.csv'):
                df = pd.read_csv('loan_predictions.csv')
                count = len(df)
                self.stats_label.config(text=f"Saved Predictions: {count}")
            else:
                self.stats_label.config(text="Saved Predictions: 0")
        except Exception as e:
            self.stats_label.config(text="Saved Predictions: Unknown")
    
    def load_sample_data(self):
        """Load sample data for testing"""
        sample_data = {
            'person_age': '35',
            'person_income': '65000',
            'person_emp_length': '5',
            'loan_amnt': '15000',
            'loan_int_rate': '12.5',
            'loan_percent_income': '0.23',
            'cb_person_cred_hist_length': '4'
        }
        
        for key, value in sample_data.items():
            if key in self.entries:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, value)
        
        # Set default values for comboboxes
        if 'person_home_ownership' in self.combos:
            self.combos['person_home_ownership'].set('MORTGAGE')
        if 'loan_intent' in self.combos:
            self.combos['loan_intent'].set('DEBTCONSOLIDATION')
        if 'loan_grade' in self.combos:
            self.combos['loan_grade'].set('B')
    
    def clear_inputs(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        for combo in self.combos.values():
            combo.set('')
        self.default_var.set(False)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def validate_inputs(self):
        """Validate all input fields"""
        # Validate numeric fields
        for feature in self.numeric_features:
            try:
                value = float(self.entries[feature].get())
                if value < 0:
                    raise ValueError(f"{feature.replace('_', ' ').title()} cannot be negative")
                # Additional validation based on your data cleaning rules
                if feature == 'person_age' and (value < 0 or value > 120):
                    raise ValueError("Age must be between 0 and 120")
                elif feature == 'person_emp_length' and value > 100:
                    raise ValueError("Employment length cannot exceed 100 years")
                elif feature == 'loan_int_rate' and value > 100:
                    raise ValueError("Interest rate cannot exceed 100%")
                elif feature == 'person_income' and value == 0:
                    raise ValueError("Income cannot be zero")
            except ValueError as e:
                if "could not convert" in str(e):
                    raise ValueError(f"Invalid value for {feature.replace('_', ' ').title()}: must be a number")
                else:
                    raise e
        
        # Validate categorical fields
        for feature in self.categorical_features:
            if not self.combos[feature].get():
                raise ValueError(f"Please select a value for {feature.replace('_', ' ').title()}")
    
    def predict_loan_status(self):
        """Main prediction function"""
        try:
            # Validate inputs
            self.validate_inputs()
            
            # Collect inputs
            data = {}
            
            # Collect numeric features
            for feature in self.numeric_features:
                value = float(self.entries[feature].get())
                data[feature] = value
            
            # Collect categorical features
            for feature in self.categorical_features:
                data[feature] = self.combos[feature].get()
            
            # Collect binary feature
            data['cb_person_default_on_file'] = 1 if self.default_var.get() else 0
            
            # Create DataFrame
            input_df = pd.DataFrame([data])
            
            # Store original values for saving
            original_values = input_df.copy()
            
            # Apply log transformation to person_income (as in training)
            input_df['person_income'] = np.log(input_df['person_income'] + 1e-10)
            
            # Handle loan_grade (replace F,G with Other as in training)
            input_df['loan_grade'] = input_df['loan_grade'].replace(['F', 'G'], 'Other')
            
            # Apply Box-Cox transformations
            for col in self.boxcox_features:
                if col in input_df.columns:
                    lambda_, shift = self.lambdas[col]
                    value = input_df[col].values[0]
                    shifted_value = value + shift
                    if shifted_value <= 0:
                        raise ValueError(f"Value for {col} after shift is non-positive: {shifted_value}")
                    transformed = stats.boxcox(shifted_value, lmbda=lambda_)
                    input_df[col] = transformed
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            proba = self.model.predict_proba(input_df)[0][1]
            
            # Format results
            risk_level = "High Risk (Default Likely)" if prediction == 1 else "Low Risk (Default Unlikely)"
            probability = f"{proba:.2%}"
            
            # Display results
            self.display_results(original_values, prediction, proba, risk_level, probability)
            
            # Save prediction
            self.save_prediction(original_values, prediction, proba)
            
            # Update prediction count
            self.update_prediction_count()
            
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
    
    def display_results(self, input_df, prediction, proba, risk_level, probability):
        """Display prediction results in the text area"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result_text = f"""
=== PREDICTION RESULT ===
Timestamp: {timestamp}

Risk Assessment: {risk_level}
Default Probability: {probability}

=== INPUT DATA ===
Age: {input_df['person_age'].iloc[0]:.0f} years
Income: ${input_df['person_income'].iloc[0]:,.0f}
Employment Length: {input_df['person_emp_length'].iloc[0]:.0f} years
Loan Amount: ${input_df['loan_amnt'].iloc[0]:,.0f}
Interest Rate: {input_df['loan_int_rate'].iloc[0]:.2f}%
Loan-to-Income Ratio: {input_df['loan_percent_income'].iloc[0]:.2%}
Credit History Length: {input_df['cb_person_cred_hist_length'].iloc[0]} years
Home Ownership: {input_df['person_home_ownership'].iloc[0]}
Loan Purpose: {input_df['loan_intent'].iloc[0]}
Loan Grade: {input_df['loan_grade'].iloc[0]}
Previous Default: {'Yes' if input_df['cb_person_default_on_file'].iloc[0] == 1 else 'No'}

=== RECOMMENDATION ===
"""
        
        if prediction == 1:
            result_text += "⚠️  HIGH RISK - Consider additional verification or reject application\n"
        else:
            result_text += "✅ LOW RISK - Application meets standard criteria\n"
        
        # Update result display
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
        
        # Show popup notification
        messagebox.showinfo("Prediction Complete", 
                           f"Risk Level: {risk_level}\nDefault Probability: {probability}")
    
    def save_prediction(self, input_df, prediction, proba):
        """Save prediction to CSV file"""
        try:
            # Prepare data for saving
            save_df = input_df.copy()
            save_df['predicted_loan_status'] = prediction
            save_df['default_probability'] = proba
            save_df['prediction_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_df['risk_level'] = "High Risk" if prediction == 1 else "Low Risk"
            
            # Save to CSV
            file_path = 'loan_predictions.csv'
            if os.path.exists(file_path):
                save_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                save_df.to_csv(file_path, mode='w', header=True, index=False)
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save prediction: {e}")
    
    def export_all_data_csv(self):
        """Export all saved predictions to CSV"""
        try:
            if not os.path.exists('loan_predictions.csv'):
                messagebox.showinfo("No Data", "No predictions have been saved yet.")
                return
            
            # Ask user where to save the file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Predictions to CSV"
            )
            
            if file_path:
                # Copy the existing data to new location
                import shutil
                shutil.copy('loan_predictions.csv', file_path)
                messagebox.showinfo("Export Complete", f"Data exported successfully to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export data: {e}")
    
    def export_all_data_pdf(self):
        """Export all saved predictions to PDF"""
        try:
            # Check if fpdf2 is installed
            try:
                from fpdf import FPDF
            except ImportError:
                messagebox.showerror("Import Error", 
                    "PDF export requires 'fpdf2' library.\nPlease install it using: pip install fpdf2")
                return
            
            if not os.path.exists('loan_predictions.csv'):
                messagebox.showinfo("No Data", "No predictions have been saved yet.")
                return
            
            # Ask user where to save the file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export Predictions to PDF"
            )
            
            if file_path:
                # Read the data
                df = pd.read_csv('loan_predictions.csv')
                
                # Create PDF
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                
                # Title
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Credit Risk Predictions Report", ln=True, align="C")
                pdf.ln(5)
                
                # Timestamp
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.ln(5)
                
                # Total records
                pdf.cell(0, 10, f"Total Records: {len(df)}", ln=True)
                pdf.ln(10)
                
                # Table headers
                pdf.set_font("Arial", "B", 8)
                headers = ['Timestamp', 'Age', 'Income', 'Loan Amount', 'Risk Level', 'Probability']
                col_width = pdf.w / len(headers) - 5
                
                for header in headers:
                    pdf.cell(col_width, 10, header, border=1, align="C")
                pdf.ln()
                
                # Table data
                pdf.set_font("Arial", "", 8)
                for index, row in df.iterrows():
                    try:
                        pdf.cell(col_width, 8, str(row.get('prediction_timestamp', 'N/A'))[:15], border=1)
                        pdf.cell(col_width, 8, str(row.get('person_age', 'N/A')), border=1, align="C")
                        pdf.cell(col_width, 8, f"${row.get('person_income', 0):,.0f}", border=1, align="R")
                        pdf.cell(col_width, 8, f"${row.get('loan_amnt', 0):,.0f}", border=1, align="R")
                        pdf.cell(col_width, 8, str(row.get('risk_level', 'N/A')), border=1, align="C")
                        pdf.cell(col_width, 8, f"{row.get('default_probability', 0):.1%}", border=1, align="C")
                        pdf.ln()
                    except:
                        continue
                
                # Save PDF
                pdf.output(file_path)
                messagebox.showinfo("Export Complete", f"Data exported successfully to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export data to PDF: {e}")
    
    def clear_all_predictions(self):
        """Clear all saved predictions"""
        try:
            if not os.path.exists('loan_predictions.csv'):
                messagebox.showinfo("No Data", "No predictions to clear.")
                return
            
            # Confirm with user
            result = messagebox.askyesno(
                "Confirm Clear", 
                "Are you sure you want to delete ALL saved predictions?\nThis action cannot be undone.",
                icon='warning'
            )
            
            if result:
                # Delete the file
                os.remove('loan_predictions.csv')
                
                # Update display
                self.update_prediction_count()
                messagebox.showinfo("Success", "All predictions have been cleared.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear predictions: {e}")

def create_lambdas_file():
    """Create a sample lambdas.pkl file if it doesn't exist"""
    if not os.path.exists('lambdas.pkl'):
        # These are example values - replace with your actual Box-Cox parameters
        lambdas = {
            'person_age': (0.5, 0),
            'person_emp_length': (0.3, 1),
            'loan_amnt': (0.2, 0),
            'loan_int_rate': (0.1, 0),
            'loan_percent_income': (0.4, 0),
            'cb_person_cred_hist_length': (0.6, 0)
        }
        joblib.dump(lambdas, 'lambdas.pkl')
        print("Created sample lambdas.pkl file")

def main():
    # Create lambdas file if it doesn't exist
    create_lambdas_file()
    
    root = tk.Tk()
    app = CreditRiskPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()