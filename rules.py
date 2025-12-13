from datetime import datetime

def validate_leave_request(employment_status, service_period, leave_duration, 
                           leave_type, start_date, end_date, reason):
    """
    Apply rule-based validation for leave requests.
    
    Returns:
        dict: {
            'passed': bool,
            'message': str,
            'auto_approved': bool (optional)
        }
    """
    
    # Rule 1: End date must be >= Start date
    if end_date < start_date:
        return {
            'passed': False,
            'message': 'End date cannot be before start date.'
        }
    
    # Rule 2: Reason must be meaningful (at least 10 characters)
    if len(reason.strip()) < 10:
        return {
            'passed': False,
            'message': 'Leave reason is too short. Please provide a detailed reason (minimum 10 characters).'
        }
    
    # Rule 3: Service period must be at least 30 days
    if service_period < 30:
        return {
            'passed': False,
            'message': f'Employee must complete at least 30 days of service. Current service: {service_period} days.'
        }
    
    # Rule 4: Maximum leave duration is 15 days
    if leave_duration > 15:
        return {
            'passed': False,
            'message': f'Leave duration exceeds maximum allowed (15 days). Requested: {leave_duration} days.'
        }
    
    # Rule 5: Contract employees cannot request more than 5 days
    if employment_status == 'Contract' and leave_duration > 5:
        return {
            'passed': False,
            'message': f'Contract employees cannot request more than 5 days of leave. Requested: {leave_duration} days.'
        }
    
    # Rule 6: Probation employees cannot request more than 2 days
    if employment_status == 'Probation' and leave_duration > 2:
        return {
            'passed': False,
            'message': f'Probation employees cannot request more than 2 days of leave. Requested: {leave_duration} days.'
        }
    
    # Rule 7: Casual leave cannot exceed 3 days
    if leave_type == 'Casual' and leave_duration > 3:
        return {
            'passed': False,
            'message': f'Casual leave cannot exceed 3 days. Requested: {leave_duration} days.'
        }
    
    # Rule 8: Earned leave requires minimum 6 months service (180 days)
    if leave_type == 'Earned' and service_period < 180:
        return {
            'passed': False,
            'message': f'Earned leave requires at least 6 months of service. Current service: {service_period} days.'
        }
    
    # Rule 9: Auto-approve Emergency leave if ≤ 2 days
    if leave_type == 'Emergency' and leave_duration <= 2:
        return {
            'passed': True,
            'auto_approved': True,
            'message': f'Emergency leave of {leave_duration} day(s) auto-approved as per HR policy.'
        }
    
    # All rules passed
    return {
        'passed': True,
        'message': 'All rule-based validations passed. Proceeding to ML analysis.'
    }


def get_all_rules():
    """
    Returns a list of all HR policy rules for reference.
    """
    rules = [
        "Service period must be at least 30 days",
        "Leave duration cannot exceed 15 days",
        "Contract employees: maximum 5 days leave",
        "Probation employees: maximum 2 days leave",
        "Casual leave: maximum 3 days",
        "Earned leave: requires 6 months service",
        "End date must be after or equal to start date",
        "Leave reason must be meaningful (minimum 10 characters)",
        "Emergency leave ≤ 2 days: auto-approved"
    ]
