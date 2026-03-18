import { Text, Pressable, StyleSheet } from 'react-native';
import { Link, Href } from 'expo-router';

type NavButtonProps = {
    label: string;
    href: Href;
    primary?: boolean;
};

export default function NavButton({ label, href, primary }: NavButtonProps) {
    return (
        // As child lets link pass navigation behavior into pressable, instead of wrapping it a web like anchor
        <Link href={href} asChild>
            {/* 'pressed' is passed to function which returns an array, only gives 'styles.pressed' when 'pressed' is true. It will override earlier array objects*/}
            {/* <Pressable style={({ pressed }) => [styles.button, pressed && styles.pressed]}> */}
            <Pressable
                android_ripple={{ color: 'rgba(147, 197, 253, 0.3)' }}
                style={[styles.button, primary && styles.primaryButton]}
            >
                <Text style={[styles.text, primary && styles.primaryText]}>{label}</Text>
            </Pressable>
        </Link>
    );
}

const styles = StyleSheet.create({
    button: {
        paddingVertical: 16,
        paddingHorizontal: 24,
        backgroundColor: 'rgba(30, 58, 138, 0.35)',
        borderRadius: 12,
        alignItems: 'center',
        borderWidth: 1.5,
        borderColor: 'rgba(147, 197, 253, 0.25)',
        elevation: 2,
        shadowColor: '#1E3A8A',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
    },
    primaryButton: {
        backgroundColor: 'rgba(59, 130, 246, 0.25)',
        borderColor: 'rgba(96, 165, 250, 0.6)',
        elevation: 4,
        shadowOpacity: 0.4,
    },
    text: {
        fontSize: 15,
        fontWeight: '600',
        color: '#94A3B8',
        letterSpacing: 0.5,
    },
    primaryText: {
        color: '#93C5FD',
        fontWeight: '700',
        fontSize: 16,
    },
});
